import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Launching inference")
    parse.add_argument('--model_path', type=str, help='Path to model weight')
    parse.add_argument('--test_data_path', type=str, help='Path to test data')
    parse.add_argument('--save_result_path', type=str, help='Path to save the result')
    
    args = parse.parse_args()

    return args
