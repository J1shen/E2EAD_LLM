import cv2
import torch
import torch.nn as nn
import einops
from blip2 import Blip2Base,disabled_train
from llama import LlamaForCausalLM
from transformers import LlamaTokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECICSE = 'fp32'
#image reading
img = cv2.imread('test.jpg')
img_resized = cv2.resize(img, (224, 224))
image = einops.repeat(img_resized, 'h w c -> (b t) c h w',t = 1,b = 1)
image = torch.tensor(image).float()

class Mymodel(Blip2Base):
    def __init__(self,
                 img_size = 224,
                 drop_path_rate = 0,
                 use_grad_checkpoint = False, 
                 vit_precision = PRECICSE,
                 freeze_vit = True,
                 num_query_token = 32,
                 freeze_qformer = True,
                 max_frame_pos = 32,
                 num_video_query_token=32,
                 frozen_video_Qformer = True,
                 llama_model= "decapoda-research/llama-7b-hf",
                 llama_proj_model='',
                 frozen_llama_proj = True
                 ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            print("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            print("freeze Qformer")

        if vit_precision == 'fp16':
            self.Qformer.half()
        print('Loading Q-Former Done')

        print('Loading Vedio-Former')
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)
        
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            print('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            print('video_Qformer is not frozen')
        print('Loading Vedio-Former Done')
        
        print('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        print('Loading LLAMA Model')
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model,torch_dtype=torch.float16)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        print('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = model.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            print('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            print('LLAMA proj is not frozen')
        print('Loading llama_proj Done')
    
    def get_embed(self,image):
        device = image.device
        image_embeds = self.visual_encoder(image)
        image_embeds_normed = self.ln_vision(image_embeds).to(device)
        return image_embeds_normed
    
    def get_token(self,image):
        device = image.device
        with self.maybe_autocast():
            self.visual_encoder.to(device)
            self.ln_vision.to(device)
            self.Qformer.to(device)
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
            query_output = self.Qformer.bert(
                query_embeds= query_tokens,
                encoder_hidden_states= image_embeds,
                encoder_attention_mask= image_atts,
                return_dict=True,
            )

        return query_output
    
    def get_llm_input(self,image):
        device = image.device
        batch_size = time_length = 1
        with self.maybe_autocast():

            self.visual_encoder.to(device)
            self.ln_vision.to(device)
            self.Qformer.to(device)
            self.video_Qformer.to(device)
            self.video_frame_position_embedding.to(device)
            self.llama_proj.to(device)
            self.video_query_tokens.to(device)

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
            query_output = self.Qformer.bert(
                query_embeds= query_tokens,
                encoder_hidden_states= image_embeds,
                encoder_attention_mask= image_atts,
                return_dict=True,
            )
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length).to(device)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama,atts_llama


model = Mymodel()
image = image.to(device).half() if PRECICSE == 'fp16' else image.to(device)

#embedding = model.get_embed(image)
#token = model.get_token(image)
inputs_llama,atts_llama = model.get_llm_input(image)
print(inputs_llama,atts_llama)