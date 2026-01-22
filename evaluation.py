import torch
from tqdm import tqdm
from utils import calculate_metrics

def evaluate(model, dataloader, tokenizer, sefeature_exractor, criterion, device, target_metric):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            waveforms_ac = batch['waveforms'].to(device)
            waveforms_se = batch['waveforms_se'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            token_lengths_se = batch['token_lengths_se'].to(device)
        
            labels = batch['labels'][target_metric].to(device)

            latents, commits, codes = tokenizer(waveforms_ac)

            whispers = sefeature_exractor(waveforms_se)
            
            outputs = model(codes, latents, commits, whispers, token_lengths, token_lengths_se)

            if target_metric == 'quality':
                predictions = outputs['quality']
            else:
                predictions = outputs['intell']

            loss_avg = criterion(predictions, labels)
            loss = loss_avg
            total_loss += loss.item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics, all_preds, all_labels