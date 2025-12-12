library(GSVA)
omics_file <- "../PCGraph-DRP/data/raw/reduced_omics_data.csv
pathway_file <- "../PCGraph-DRP/data/raw/filtered_pathways_714.csv"

output_dir <- "E:/git bash file/drug_respones2/BANDRP_network"
output_file <- file.path(output_dir, "pathway_activity_ssgsea_BANDRP.csv")

print("步骤 1: 正在读取数据文件...")
tryCatch({
  omics_df <- read.csv(omics_file, stringsAsFactors = FALSE, check.names = FALSE)
  pathway_df <- read.csv(pathway_file, stringsAsFactors = FALSE, header = TRUE)
}, error = function(e) {
  stop(paste("错误：无法读取输入文件，请检查路径。原始错误信息:", e))
})

colnames(pathway_df)[1:2] <- c("genes", "name")
print(paste("组学数据维度:", nrow(omics_df), "x", ncol(omics_df)))
print(paste("通路数据维度:", nrow(pathway_df), "x", ncol(pathway_df)))

print("步骤 2: 正在准备基因表达矩阵...")
exp_cols <- grep("_exp$", colnames(omics_df), value = TRUE)
if (length(exp_cols) == 0) {
  stop("错误：在组学数据中未找到任何以'_exp'结尾的列。")
}
print(paste("找到", length(exp_cols), "个基因的表达数据。"))
cell_line_names <- omics_df[, 1]
exp_data <- omics_df[, exp_cols]
colnames(exp_data) <- gsub("_exp$", "", colnames(exp_data))
rownames(exp_data) <- cell_line_names
expr_matrix <- t(as.matrix(exp_data))
print(paste("表达矩阵维度:", nrow(expr_matrix), "基因 x", ncol(expr_matrix), "细胞系"))

print("步骤 3: 正在准备通路列表...")
pathway_list <- list()
for (i in 1:nrow(pathway_df)) {
  pathway_name <- pathway_df$name[i]
  if (is.na(pathway_name) || pathway_name == "") {
    next 
  }
  genes <- strsplit(as.character(pathway_df$genes[i]), "\\|")[[1]]
  genes <- genes[genes != ""]
  if (length(genes) > 0) {
    pathway_list[[pathway_name]] <- genes
  }
}
print(paste("成功处理", length(pathway_list), "个通路。"))


# ===== 4. 运行 ssGSEA 计算 (已修改) =====
print("步骤 4: 正在计算 ssGSEA 通路活性分数...")

# --- MODIFIED: 改为先创建参数对象，再调用gsva ---
# 这是更稳定和兼容性更强的用法
ssgsea_param <- ssgseaParam(
  exprData = expr_matrix,      # 表达矩阵
  geneSets = pathway_list,     # 通路-基因列表
  minSize = 1                  # 最小通路基因数
)

# 使用参数对象调用gsva函数
gsva_scores <- gsva(param = ssgsea_param, verbose = TRUE)
# --------------------------------------------------

print("ssGSEA 计算完成！")


# ===== 5. 整理并保存结果 =====
print("步骤 5: 正在整理并保存结果...")
pathway_activity_df <- as.data.frame(t(gsva_scores))
pathway_activity_df <- cbind(cell_line = rownames(pathway_activity_df), pathway_activity_df)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(pathway_activity_df, output_file, row.names = FALSE)

print("--- 任务完成 ---")
print(paste("最终结果维度:", nrow(pathway_activity_df), "细胞系 x", (ncol(pathway_activity_df)-1), "通路"))
print("最终文件已保存至:")
print(output_file)
print("结果预览 (前5行，前4列):")
print(head(pathway_activity_df[, 1:4]))