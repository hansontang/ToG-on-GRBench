import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="grbench", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="/home/hansont2/ToG/ToG/ToG_grbench.jsonl", help="the output file name.")
    parser.add_argument("--method", type=str, 
                        default="ToG", help="The name of the method being evaluated.")
    args = parser.parse_args()

    # prepare_dataset_for_eval is now corrected to load JSONL files
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    num_right = 0
    num_error = 0
    for data in output_datas:
        answers = align(args.dataset, question_string, data, ground_truth_datas)
        results = data['results']
        
        # 1. 首先，尝试用智能的方式提取大括号里的内容
        response = extract_content(results)

        # 2. 如果提取失败（因为没有大括号），则启用“备用方案”
        if response == "NULL":
            # 将模型的完整原始回答作为备选答案
            response = results

        # 3. 使用宽容的匹配函数进行比较
        if exact_match(response, answers):
            num_right += 1
        else:
            num_error += 1
    
    # This part prints to the console
    print(f"Total Processed: {len(output_datas)}")
    print(f"Right: {num_right}, Error: {num_error}")
    if len(output_datas) > 0:
        print(f"Exact Match: {float(num_right / len(output_datas))}")

    # Save the final results to a JSON file
    save_result2json(args.dataset, num_right, num_error, len(output_datas), args.method)
    
    print(f"\nEvaluation results saved to ToG_{args.dataset}_results.json")