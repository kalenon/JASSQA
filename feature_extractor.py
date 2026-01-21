import torch
import joblib
from transformers import EncodecModel, HubertModel, AutoFeatureExtractor, WhisperModel
import logging
from utils import encode_and_stack_layers
import os

class BaseTokenizer:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.sampling_rate = None
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def __call__(self, waveforms):
        raise NotImplementedError
    
class DACTokenizer(BaseTokenizer):
    def _load_model(self):
        import dac
        self.model_path = dac.utils.download(model_type=self.model_path)
        logging.info(f"Loading DAC: {self.model_path}")
        self.model = dac.DAC.load(self.model_path).to(self.device)
        self.sampling_rate = self.model.sample_rate

    @torch.no_grad()
    def __call__(self, waveforms):
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        x = self.model.preprocess(waveforms, self.sampling_rate)

        z, codes, latents, commit_loss = encode_and_stack_layers(self.model, x)

        return latents, commit_loss, codes

class WhisperFeatureExtractor_largev3(BaseTokenizer):
    def _load_model(self):
        logging.info(f"Loading Whisper: openai/whisper-large-v3")
        self.embed_dim = 1280
        self.model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")

        self.sampling_rate = self.processor.sampling_rate
        self.model.eval() 

    @torch.no_grad()
    def __call__(self, waveforms: torch.Tensor):
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        
        inputs = self.processor(
            waveforms.cpu().numpy(), 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)

        encoder_outputs = self.model.encoder(input_features)
        
        return encoder_outputs.last_hidden_state
    
class WhisperFeatureExtractor_medium(BaseTokenizer):
    def _load_model(self):

        logging.info(f"Loading Whisper: openai/whisper-medium")
        self.embed_dim = 1024
        self.model = WhisperModel.from_pretrained("openai/whisper-medium").to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")

        self.sampling_rate = 16000
        self.model.eval()

    @torch.no_grad()
    def __call__(self, waveforms: torch.Tensor):
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        
        inputs = self.processor(
            waveforms.cpu().numpy(), 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device)

        encoder_outputs = self.model.encoder(input_features)
        
        return encoder_outputs.last_hidden_state