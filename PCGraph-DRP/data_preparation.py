import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

# --- 配置参数 (请在此处修改) ---

# 1. 输入的三元组文件路径
TRIPLETS_FILE_PATH = r"E:\git bash file\drug_respones2\data\processed\drug_cell_ic50_triples.csv"

# 2. 输出划分结果的目录 (脚本会自动创建此目录)
OUTPUT_SPLITS_DIR = r"E:\git bash file\drug_respones2\data\splits"

SPLIT_METHOD = 'random'

TEST_SIZE = 0.2           
VAL_TEST_SPLIT = 0.5
RANDOM_SEED = 42

def load_triplets_data(file_path):
    """
    直接从指定路径加载三元组数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：找不到输入文件，请检查路径: {file_path}")
        
    # 加载三元组数据
    triplets_df = pd.read_csv(file_path)
    
    # 重置索引，确保索引从0开始连续
    triplets_df = triplets_df.reset_index(drop=True)
    
    print(f"成功加载了 {len(triplets_df)} 个三元组，来自: {file_path}")
    return triplets_df

def analyze_coverage(data, data_type, all_cell_lines, all_drugs):
    """分析数据集的细胞系和药物覆盖情况"""
    if data.empty:
        print(f"警告: {data_type} 数据集为空，无法进行覆盖分析")
        return
    
    # 提取数据中的唯一细胞系和药物
    unique_cells = data['cell_line_id'].unique()
    unique_drugs = data['drug_id'].unique()
    
    # 计算覆盖比例
    cell_coverage = len(unique_cells) / len(all_cell_lines) * 100
    drug_coverage = len(unique_drugs) / len(all_drugs) * 100
    
    print(f"\n=== {data_type} 数据集覆盖分析 ===")
    print(f"数据集样本数: {len(data)}")
    print(f"唯一细胞系数量: {len(unique_cells)} / {len(all_cell_lines)} ({cell_coverage:.2f}%)")
    print(f"唯一药物数量: {len(unique_drugs)} / {len(all_drugs)} ({drug_coverage:.2f}%)")
    print(f"细胞系列表示例: {unique_cells[:5].tolist() + (['...'] if len(unique_cells) > 5 else [])}")
    print(f"药物ID列表示例: {unique_drugs[:5].tolist() + (['...'] if len(unique_drugs) > 5 else [])}")

def split_dataset(triplets_df, output_dir, split_method, test_size, val_split, seed):
    """
    划分训练、验证、测试集，并保存结果
    """
    if split_method not in ['cell-based', 'random']:
        print(f"警告: 不支持的划分方法 '{split_method}'，将使用 'random' 方法")
        split_method = 'random'
    
    os.makedirs(output_dir, exist_ok=True)
    triplets_df = triplets_df.reset_index(drop=True)
    
    # 提前获取全局唯一细胞系和药物（基于原始数据）
    all_cell_lines = triplets_df['cell_line_id'].unique()
    all_drugs = triplets_df['drug_id'].unique()
    print(f"\n全局唯一细胞系数量: {len(all_cell_lines)}")
    print(f"全局唯一药物数量: {len(all_drugs)}")

    if split_method == 'cell-based':
        print("\n使用【按细胞系】划分方法...")
        all_cell_lines_list = sorted(list(all_cell_lines))
        print(f"总共有 {len(all_cell_lines_list)} 个唯一的细胞系")

        # 划分细胞系
        train_cells, temp_cells = train_test_split(
            all_cell_lines_list, 
            test_size=test_size, 
            random_state=seed
        )

        val_cells, test_cells = train_test_split(
            temp_cells, 
            test_size=val_split, 
            random_state=seed
        )

        print(f"训练集细胞系数量: {len(train_cells)}")
        print(f"验证集细胞系数量: {len(val_cells)}")
        print(f"测试集细胞系数量: {len(test_cells)}")

        # 按细胞系筛选数据
        train_data = triplets_df[triplets_df['cell_line_id'].isin(train_cells)]
        val_data = triplets_df[triplets_df['cell_line_id'].isin(val_cells)]
        test_data = triplets_df[triplets_df['cell_line_id'].isin(test_cells)]
        
        suffix = "cell_based"
        
    elif split_method == 'random':
        print("\n使用【随机】划分方法...")
        all_indices = triplets_df.index.tolist()
        
        # 随机划分索引
        train_indices, temp_indices = train_test_split(
            all_indices, 
            test_size=test_size, 
            random_state=seed
        )

        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=val_split, 
            random_state=seed
        )
        
        # 按索引获取数据
        train_data = triplets_df.loc[train_indices].reset_index(drop=True)
        val_data = triplets_df.loc[val_indices].reset_index(drop=True)
        test_data = triplets_df.loc[test_indices].reset_index(drop=True)
        
        suffix = "random"
    
    print(f"\n训练集三元组数量: {len(train_data)}")
    print(f"验证集三元-组数量: {len(val_data)}")
    print(f"测试集三元组数量: {len(test_data)}")
    
    # --- 保存文件 ---
    # 1. 保存各数据集包含的细胞系ID
    pd.DataFrame({'cell_line_id': train_data['cell_line_id'].unique()}).to_csv(os.path.join(output_dir, f'train_cells_{suffix}.csv'), index=False)
    pd.DataFrame({'cell_line_id': val_data['cell_line_id'].unique()}).to_csv(os.path.join(output_dir, f'val_cells_{suffix}.csv'), index=False)
    pd.DataFrame({'cell_line_id': test_data['cell_line_id'].unique()}).to_csv(os.path.join(output_dir, f'test_cells_{suffix}.csv'), index=False)
    
    # 2. 保存完整的三元组数据
    train_data.to_csv(os.path.join(output_dir, f'train_data_{suffix}.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, f'val_data_{suffix}.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, f'test_data_{suffix}.csv'), index=False)
    
    print(f"\n所有划分结果已保存至目录: {output_dir}")
    
    # --- 覆盖分析 ---
    analyze_coverage(train_data, "训练集", all_cell_lines, all_drugs)
    analyze_coverage(val_data, "验证集", all_cell_lines, all_drugs)
    analyze_coverage(test_data, "测试集", all_cell_lines, all_drugs)
    
    return train_data, val_data, test_data

def main():
    """主函数，执行数据加载和划分流程"""
    try:
        # 使用顶部配置的参数
        triplets_df = load_triplets_data(TRIPLETS_FILE_PATH)
        train_data, val_data, test_data = split_dataset(
            triplets_df=triplets_df,
            output_dir=OUTPUT_SPLITS_DIR,
            split_method=SPLIT_METHOD,
            test_size=TEST_SIZE,
            val_split=VAL_TEST_SPLIT,
            seed=RANDOM_SEED
        )
        
        print("\n数据划分全部完成。")
        print(f"训练集: {len(train_data)} 个样本")
        print(f"验证集: {len(val_data)} 个样本")
        print(f"测试集: {len(test_data)} 个样本")
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()