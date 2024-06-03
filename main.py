from config import parse_args
from inference import run_inference

def main():
    args = parse_args()
    model_weights_path = args.model_path
    test_data_path = args.test_data_path
    save_result_path = args.save_result_path

    run_inference(model_weights_path, test_data_path, save_result_path)

if __name__ == '__main__':
    main()
