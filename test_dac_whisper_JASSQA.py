import argparse
import numpy as np
import torch
import logging
import os
import pandas as pd

from evaluation import evaluate
from utils import setup_logging, plot_scatterplot
from data_loader_online import get_dataloader, MyCollator
from feature_extractor import DACTokenizer, WhisperFeatureExtractor_largev3, WhisperFeatureExtractor_medium
from model import MosPredictor_mosanetplus_scoreq_crossatt_8

def main(args):
    save_dir = args.checkpoint_path[:-3] + args.audio_root_dir.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(os.path.join(save_dir, 'test_run.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Use of equipment: {device}")
    logging.info(f"Test parameters: {args}")

    if args.ac_tokenizer_type == 'dac':
        tokenizer = DACTokenizer("44khz", device=device)
        downsampling_factor = tokenizer.model.hop_length
    else:
        raise ValueError(f"Unsupported ac feature extractor types: {args.ac_tokenizer_type}")
    tokenizer.model.eval()

    if args.sematic_type == 'whisper_medium':
        sefeature_exractor = WhisperFeatureExtractor_medium(None, device=device)
        embed_dim = sefeature_exractor.embed_dim
        downsampling_factor_se = 320
    elif args.sematic_type == 'whisper_largev3':
        sefeature_exractor = WhisperFeatureExtractor_largev3(None, device=device)
        embed_dim = sefeature_exractor.embed_dim
        downsampling_factor_se = 320
    elif args.sematic_type == 'w2v_large':
        embed_dim = 1024   

    target_sr = tokenizer.sampling_rate
    target_sr_se = sefeature_exractor.sampling_rate
    logging.info("Test data is currently being added....")
    collator = MyCollator(downsampling_factor, downsampling_factor_se)
    test_loader = get_dataloader(
        dataset_path_or_name=args.test_dataset, 
        split='test',
        target_sr=target_sr, 
        target_sr_se=target_sr_se,
        batch_size=args.batch_size, 
        shuffle=False, 
        audio_root_dir=args.audio_root_dir,
        collate_fn=collator
    )
    
    if test_loader is None:
        logging.error("Unable to create the test data loader; the programme has exited.")
        return


    if args.ac_tokenizer_type == 'dac':
        vocab_size=tokenizer.model.codebook_size
        vocab_dim=tokenizer.model.codebook_dim
        n_codebook=tokenizer.model.n_codebooks
    else:
        vocab_size = 1024  

    model = MosPredictor_mosanetplus_scoreq_crossatt_8(vocab_size=vocab_size, vocab_dim=vocab_dim, n_codebook=n_codebook, embed_dim=embed_dim)
    model.to(device)
    logging.info(f"Loading ckpt weights from {args.checkpoint_path} ...")
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    logging.info("Commencing testing...")
    criterion = torch.nn.MSELoss()
    loss, metrics, predictions, labels = evaluate(model, test_loader, tokenizer, sefeature_exractor, criterion, device, args.target_metric)
    
    if metrics:
        logging.info("Test complete.")
        logging.info(f"Test set Loss={loss}")
        logging.info(f"Test set pred MOS: {np.mean(predictions)}")
        logging.info(f"Test set metrics: SRCC={metrics.get('SRCC', 'N/A'):.4f}, LCC={metrics.get('LCC', 'N/A'):.4f}, MSE={metrics.get('MSE', 'N/A'):.4f}")
    else:
        logging.info("Test complete (no labels).")

    file_name = os.path.basename(args.test_dataset)[:-4]
    if args.output_csv:
        results_df = pd.DataFrame({
            'filepath_deg': test_loader.dataset.df[test_loader.dataset.audio_col],
            'prediction': predictions,
            'ground_truth': labels if labels else ['N/A'] * len(predictions)
        })
        output_path = os.path.join(save_dir, file_name+'.csv')
        results_df.to_csv(output_path, index=False)
        output_path_txt = os.path.join(save_dir, file_name+'.txt')
        results_df.to_csv(output_path_txt, index=False, header=False)
        logging.info(f"Predictions saved to: {output_path}")

    if args.plot and not torch.isnan(torch.tensor(labels)).all():
        plot_path = os.path.join(save_dir, file_name+'.png')
        plot_scatterplot(predictions, labels, metrics, plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained model")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file of the trained model (.pt).")
    parser.add_argument("--ac_tokenizer_type", type=str, required=True, help="The type of acoustic feature extractor to be used.")
    parser.add_argument("--sematic_type", type=str, required=True, help="The type of semantic feature extractor to be used.")

    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--audio_root_dir", type=str, default=None, help="The root directory for audio files.")
    
    parser.add_argument("--target_metric", type=str, required=True, choices=['quality', 'intelligibility'], help="The target metric for which the model was trained to predict.")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during evaluation.")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV filename for storing prediction results (optional).")
    parser.add_argument("--plot", type=str, help="Generate and save a scatter plot of predictions vs. ground truth.")

    args = parser.parse_args()
    main(args)