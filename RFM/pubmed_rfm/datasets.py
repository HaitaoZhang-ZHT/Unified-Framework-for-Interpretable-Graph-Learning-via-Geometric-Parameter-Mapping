import os
import requests
import tarfile
from tqdm import tqdm


def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("下载错误")


def download_cora_dataset(data_dir='./data'):
    """下载并解压Cora数据集"""
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 数据集URL
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    tar_filename = os.path.join(data_dir, "cora.tgz")
    extracted_dir = os.path.join(data_dir, "cora")

    # 下载数据集
    if not os.path.exists(tar_filename):
        print(f"开始下载Cora数据集到 {tar_filename}")
        download_file(url, tar_filename)
        print("下载完成")
    else:
        print(f"数据集已存在: {tar_filename}")

    # 解压数据集
    if not os.path.exists(extracted_dir):
        print(f"开始解压到 {extracted_dir}")
        with tarfile.open(tar_filename, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("解压完成")

        # 重命名解压后的目录
        cora_dir = os.path.join(data_dir, "cora")
        if not os.path.exists(cora_dir):
            os.rename(os.path.join(data_dir, "cora"), cora_dir)
    else:
        print(f"数据集已解压: {extracted_dir}")

    # 验证文件
    content_file = os.path.join(extracted_dir, "cora.content")
    cites_file = os.path.join(extracted_dir, "cora.cites")

    if os.path.exists(content_file) and os.path.exists(cites_file):
        print("数据集文件验证成功:")
        print(f"- {content_file}")
        print(f"- {cites_file}")
        return content_file, cites_file
    else:
        print("错误: 找不到必要的数据集文件")
        return None, None


if __name__ == "__main__":
    content_file, cites_file = download_cora_dataset()
    if content_file and cites_file:
        print("\nCora数据集已准备就绪!")