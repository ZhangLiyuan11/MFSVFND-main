import torch

from model.cross_attention_module import CrossTransformer
from model.modality_attention_module import ModalityTransformer
from utils.tools import *

class classifier(nn.Module):
    def __init__(self, fea_dim, dropout_probability):
        super(classifier, self).__init__()
        self.class_net = nn.Sequential(
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(32, 2)
        )

    def forward(self, fea):
        out = self.class_net(fea)
        return out
    

class MFSVFNDModel(torch.nn.Module):
    def __init__(self, fea_dim, dropout, dataset):
        super(MFSVFNDModel, self).__init__()
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768

        self.img_dim = 1024
        self.audio_dim = 1024

        self.dim = fea_dim
        self.num_heads = 8
        self.trans_dim = 512
        self.dropout = dropout

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        # text
        self.text_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.text_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)

        # image
        self.image_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.image_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)

        # audio
        self.audio_causal_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)
        self.audio_transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                                                  dropout=self.dropout)


        # CrossTransformer
        self.tv_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.ta_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.av_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.vt_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.at_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)
        self.va_cross_transformer = CrossTransformer(model_dimension=512, number_of_heads=8, dropout_probability=self.dropout)


        self.classifier = classifier(fea_dim=512, dropout_probability=self.dropout)

    def forward(self, **kwargs):
        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512,D)
        title_mask = kwargs['title_mask']  
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  
        fea_text = self.linear_text(fea_text)

        ### Audio Frames ###
        fea_audio = kwargs['audio_feas']  # (B,L,D)
        fea_audio = self.linear_audio(fea_audio)

        ### Image Frames ###
        frames = kwargs['frames']  # (B,L,D)
        fea_image = self.linear_img(frames)

        # text
        fea_text = self.text_causal_transformer(fea_text, is_causal=True)
        fea_text = self.text_transformer(fea_text)

        # image
        fea_image = self.image_causal_transformer(fea_image, is_causal=True)
        fea_image = self.image_transformer(fea_image)

        # audio
        fea_audio = self.audio_causal_transformer(fea_audio, is_causal=True)
        fea_audio = self.audio_transformer(fea_audio)

        # cross_attention
        fea_tv = self.tv_cross_transformer(fea_text, fea_image)
        fea_vt = self.vt_cross_transformer(fea_image, fea_text)

        fea_ta = self.ta_cross_transformer(fea_text, fea_audio)
        fea_at = self.at_cross_transformer(fea_audio, fea_text)

        fea_va = self.va_cross_transformer(fea_image, fea_audio)
        fea_av = self.av_cross_transformer(fea_audio, fea_image)
        
        fea_tv = torch.mean(torch.cat((fea_tv,fea_vt),dim=1), dim=1)
        fea_ta = torch.mean(torch.cat((fea_ta,fea_at),dim=1), dim=1)
        fea_va = torch.mean(torch.cat((fea_va,fea_av),dim=1), dim=1)

        final_fea = fea_tv + fea_ta + fea_va
        
        output = self.classifier(final_fea)

        return output
