import os
import time
from contextlib import redirect_stdout
from pathlib import Path
import io
import torch
from transformers import AutoTokenizer
from utils.model_utils import get_model
from utils.data_utils import get_calib_data
from utils.eval_utils import evaluate
from utils.process_args import get_boa_arguments, get_boa_weight_quant_infos
from quantize import boa_fwrd

if __name__ == '__main__':
    args = get_boa_arguments()

    # configure CPU threading before any heavy tensor work
    if args.num_cpu_threads is not None:
        torch.set_num_threads(args.num_cpu_threads)
        os.environ.setdefault('OMP_NUM_THREADS', str(args.num_cpu_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(args.num_cpu_threads))
        print(f"[threading] torch num_threads={args.num_cpu_threads}, "
              f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
              f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}")
    if args.num_workers > 1:
        print(f"[threading] layer-parallel workers={args.num_workers}")

    # load model
    with redirect_stdout(io.StringIO()) as f:
        llm = get_model(args.llm_path)
    llm.seqlen = args.seqlen
    llm.eval()

    # evaluate the fp model performance
    if args.eval_fp:
        results = evaluate(llm, args)
        print(results)
        exit(0)

    # load calib. data
    calib_data = get_calib_data(args)

    # quantize
    qconfigs, boa_opts, hyperparams = get_boa_weight_quant_infos(args)
    print("Start quantization")
    tick = time.time()
    boa_fwrd(llm, calib_data, qconfigs, boa_opts, hyperparams, args)
    process_time = round(time.time() - tick, 3)
    print(f"Quantization processing time: {process_time}")

    # evaluate
    print(args)
    results = evaluate(llm, args)
    results['time'] = process_time
    print(results)

    # save fake-quantized model and tokenizer
    if args.save_path is not None:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving fake-quantized model to: {save_dir}")
        llm.save_pretrained(str(save_dir))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        tokenizer.save_pretrained(str(save_dir))
        print(f"Model and tokenizer saved to: {save_dir}")