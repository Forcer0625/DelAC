import json
import csv
from io import StringIO
import numpy as np
from scipy.special import kl_div

def extract_strategy_from_csv(file_name="example.csv"):    
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        csv_line = next(csv.reader(StringIO(lines[20])))

        # 不去重，直接取第 2、3 項
        selected_vectors = csv_line[1:3]
        parsed_vectors = []

        for vec in selected_vectors:
            cleaned = vec.strip('"')  # 去除外層雙引號
            parsed = [float(x.strip()) for x in cleaned.strip("[]").split(",")]
            parsed_vectors.append(parsed)

        strategy = np.array(parsed_vectors)

    return strategy

def extract_strategy_from_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    # 提取 strategy 並轉成 float，之後轉成 numpy array
    strategies = np.array([
        list(map(float, data["team 1 strategy"])),
        list(map(float, data["team 2 strategy"]))
    ])

    return strategies

def calculate_kl_div(nash_strategy, json_strategy):
    return kl_div(nash_strategy, json_strategy).sum()

if __name__ == '__main__':
    runs_data_path = 'C:/Users/yhes9/old_runs/'
    run_case = '250329-YF_GeneralSum'
    run_env = run_case[7:]
    nash_file_postfix = '-nash/data.csv'
    algos = ['cfac', 'ia2c', 'ca2c']
    
    datas = {}
    for algo_name in algos:
        datas[algo_name] = []
    
    for n_test in range(1, 31):
        csv_file_name = runs_data_path+run_case+'/'+run_env+str(n_test).zfill(3)+nash_file_postfix
        nash_strategy = extract_strategy_from_csv(csv_file_name)

        for algo_name in algos:
            algo_file_postfix = '-'+algo_name+'/config.json'
            json_file_name = runs_data_path+run_case+'/'+run_env+str(n_test).zfill(3)+algo_file_postfix
            algo_strategy = extract_strategy_from_json(json_file_name)
            datas[algo_name].append(calculate_kl_div(nash_strategy[0], algo_strategy[0]))
            datas[algo_name].append(calculate_kl_div(nash_strategy[1], algo_strategy[1]))

    print(run_env)
    for algo_name in algos:
        data = np.array(datas[algo_name])
        print(algo_name, ':', np.mean(data).round(5), ', std:', np.std(data).round(5))