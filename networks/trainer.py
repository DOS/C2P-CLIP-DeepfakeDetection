import torch
import torch.nn as nn
from networks.base_model import BaseModel

from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model



class CLIPModel_lora(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14-336', num_classes=1, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super(CLIPModel_lora, self).__init__()
        self.model        = CLIPModel.from_pretrained(name)
        self.processor    = CLIPProcessor.from_pretrained(name)
        self.vision_tower = self.model.vision_model
        self.vision_tower.requires_grad_(False)
        self.model.text_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.model.logit_scale.requires_grad_(False)
        lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['q_proj','k_proj','v_proj'],
                lora_dropout=lora_dropout,
                bias="none",
            )
        self.vision_tower_lora = get_peft_model(self.vision_tower, lora_config)
        self.model.fc = nn.Linear( 768, num_classes )
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)


    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions   = self.model.config.output_attentions,
            output_hidden_states= self.model.config.output_hidden_states,
            return_dict         = self.model.config.use_return_dict, 
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds
    
    def encode_image(self, img):
        vision_outputs = self.vision_tower_lora(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.model.config.output_hidden_states,
            return_dict          = self.model.config.use_return_dict,
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    
    
    def forward(self, img, input_ids, attention_mask, cla=False):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')

        image_embeds = self.encode_image(img)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        classhead = self.model.fc(image_embeds)
        if cla: return classhead 
        
        text_embeds  = self.encode_text(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.model.logit_scale.exp()
        logits_per_image = logits_per_text.t()

        return logits_per_image, classhead.squeeze(1)
    
    def forward_eval(self, img):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        classhead = self.model.fc(image_embeds)
        return classhead 

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr
        self.claloss = opt.claloss
        
        self.printOne = 1
        self.model = CLIPModel_lora(name=opt.clip, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha, lora_dropout=opt.lora_dropout)

        net_params = sum(map(lambda x: x.numel(), self.model.model.parameters())) 

        print(f'Model parameters {net_params:,d}')

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            elif opt.optim == 'adamw':
                self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, weight_decay=0.05, betas=(opt.beta1, 0.999), eps=1e-8)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        self.model = nn.DataParallel(self.model).cuda()

 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.delr
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/self.delr} to {param_group["lr"]} with delr {self.delr}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input           = input[1].cuda()
        self.text            = input[2]
        self.input_ids       = input[3].cuda()
        self.attention_mask  = input[4].cuda()
        self.label = input[5].cuda().float()


    def forward(self):
        self.output, self.classhead = self.model( self.input, self.input_ids, self.attention_mask )

        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        
        caption_loss = nn.functional.cross_entropy(logits    , torch.arange(len(logits), device=logits.device))
        image_loss   = nn.functional.cross_entropy(logits.t(), torch.arange(len(logits), device=logits.device))
        return (caption_loss + image_loss) / 2.0
        
    def get_loss(self):
        return self.model.clip_loss_input(self.input, self.text)

    def optimize_parameters(self):
        self.forward()
        
        self.loss1 = sum([self.contrastive_loss(output) for output in torch.split(self.output, self.output.shape[1], dim=0)])
        self.loss2 = self.claloss * self.loss_fn(self.classhead, self.label)
        self.loss  = self.loss1 + self.loss2
        # self.loss1, self.loss2 = 0.0, 0.0
        # self.loss  = self.loss_fn(self.classhead, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

