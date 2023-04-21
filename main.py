import streamlit as st
from datasets import load_dataset
from datasets import Dataset
from PIL import Image
from io import BytesIO
import base64
from huggingface_hub import Repository
import matplotlib.pyplot as plt
import ssl
import os
import pyarrow as pa
import pickle
ssl._create_default_https_context = ssl._create_unverified_context

# 转换PIL Image为base64，便于在网页中显示
def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# 获取或创建会话状态
if "session_state" not in st.session_state:
    st.session_state.session_state = {}

# 加载数据集并存储在会话状态中
if "dataset_dict" not in st.session_state.session_state:
    dataset = load_dataset("JerryMo/image-caption-blip-for-training", split="train")
    dataset_dict = {i: {"image": example["image"], "text": example["text"], "modified": False} for i, example in enumerate(dataset)}
    st.session_state.session_state["dataset_dict"] = dataset_dict

dataset_dict = st.session_state.session_state["dataset_dict"]

# 计算已修改的图像数量
def count_modified_images(dataset_dict):
    return sum(1 for example in dataset_dict.values() if example["modified"])

# 将数据集推送到 Hugging Face 仓库
def push_to_huggingface(username, api_token, repo_name, dataset_dict):
    # 将 dataset_dict 转换为 Dataset 对象
    examples = []
    for idx, example in dataset_dict.items():
        examples.append({"image": example["image"], "text": example["text"], "index": idx})
    modified_dataset = Dataset.from_dict({"image": [e["image"] for e in examples], "text": [e["text"] for e in examples], "index": [e["index"] for e in examples]})
    
    # 使用 push_to_hub 方法将数据集推送到 Hugging Face 仓库
    remote_hub_repo = f"{username}/{repo_name}"
    modified_dataset.push_to_hub(remote_hub_repo, token=api_token, private=False)

def save_modified_dataset_as_parquet(dataset_dict, output_dir, file_name="modified_dataset.parquet"):
    # 将 dataset_dict 转换为 Dataset 对象
    examples = []
    for idx, example in dataset_dict.items():
        examples.append({"image": example["image"], "text": example["text"], "index": idx})
    modified_dataset = Dataset.from_dict({"image": [e["image"] for e in examples], "text": [e["text"] for e in examples], "index": [e["index"] for e in examples]})
    
    # 将 Dataset 对象保存为 Parquet 文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    table = pa.Table.from_pandas(modified_dataset.to_pandas())
    file_path = os.path.join(output_dir, file_name)
    pa.parquet.write_table(table, file_path)

    print(f"Modified dataset saved as Parquet file: {file_path}")

def save_all_dataset_as_csv(dataset_dict, output_dir, file_name="all_captions.csv"):
    # 将 dataset_dict 转换为 Dataset 对象
    examples = []
    for idx, example in dataset_dict.items():
        examples.append({"text": example["text"], "index": idx})
    all_dataset = Dataset.from_dict({"text": [e["text"] for e in examples], "index": [e["index"] for e in examples]})
    
    # 将 Dataset 对象保存为 CSV 文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = all_dataset.to_pandas()
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)

    print(f"All dataset saved as CSV file: {file_path}")

# 主程序
def main():
    st.title("Image Caption Annotation App")

    # 添加两列布局
    col1, col2, col3 = st.columns(3)

    # 在第一列中显示已修改的图像数量
    with col1:
        modified_count = count_modified_images(dataset_dict)
        total_count = len(dataset_dict)
        st.write(f"Modified: {modified_count} / {total_count}")

        # 绘制饼图
        plt.figure(figsize=(5, 5))
        labels = ["Modified", "Not modified"]
        sizes = [modified_count, total_count - modified_count]
        colors = ["#66b3ff", "#ff9999"]
        explode = (0.1, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        plt.axis("equal")
        st.pyplot(plt.gcf())

    # 在第二列中提供推送到 Hugging Face 仓库的功能
    with col2:
        st.write("Push to Hugging Face Repository")
        username = st.text_input("Username")
        api_token = st.text_input("API Token", type="password")
        repo_name = st.text_input("Repository Name")
        push_button = st.button("Push")

        if push_button:
            if not (username and api_token and repo_name):
                st.error("Please fill in all the fields.")
            else:
                try:
                    push_to_huggingface(username, api_token, repo_name, dataset_dict)
                    st.success("Dataset pushed to Hugging Face repository successfully!")
                except Exception as e:
                    st.error(f"An error occurred while pushing the dataset: {e}")
    with col3:
        st.write("Save as Local CSV File")
        output_dir = st.text_input("Output Directory")
        save_button = st.button("Save")

        if save_button:
            if not output_dir:
                st.error("Please enter an output directory.")
            else:
                try:
                    save_all_dataset_as_csv(dataset_dict, output_dir)
                    st.success("Dataset saved as csv file successfully!")
                except Exception as e:
                    st.error(f"An error occurred while saving the dataset: {e}")

    # 显示图片和表单
    image_index = st.number_input("Select an image", 0, len(dataset_dict) - 1, 0, 1)
    st.write(f"Image {image_index + 1} of {len(dataset_dict)}")

    with st.form("annotation_form"):
        # 显示图片
        example = dataset_dict[image_index]
        image = example["image"]
        st.image(get_image_base64(image), caption="Image", use_column_width=True)

        # 显示并编辑文本
        original_text = example["text"]
        edited_text = st.text_area("Caption", value=original_text, height=200)

        # 显示当前状态
        if example.get("modified", False):
            st.warning("This image has been modified.")
        else:
            st.info("This image has not been modified.")

        # 提交按钮
        submit_button = st.form_submit_button("Submit")

        # 当点击提交按钮时，保存编辑过的文本
        if submit_button:
            if edited_text != original_text:
                dataset_dict[image_index]["text"] = edited_text
                dataset_dict[image_index]["modified"] = True
                st.session_state.session_state["dataset_dict"] = dataset_dict
                st.success("Annotation updated successfully!")

if __name__ == "__main__":
    main()
