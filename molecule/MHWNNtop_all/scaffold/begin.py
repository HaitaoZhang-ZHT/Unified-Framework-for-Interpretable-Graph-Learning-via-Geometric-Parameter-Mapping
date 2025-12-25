import subprocess
import sys
from datetime import datetime


def run_main_multiple_times(start_seed: int = 42, n_runs: int = 10):
    """
    循环运行main.py，每次递增seed和split-seed

    Args:
        start_seed: 起始种子值（默认42）
        n_runs: 运行次数（默认10次，对应seed 42~51）
    """
    # 记录整体开始时间
    start_time = datetime.now()
    print(f"=== 开始批量运行main.py | 起始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"运行次数：{n_runs}次 | 起始seed/split-seed：{start_seed} | 结束seed/split-seed：{start_seed + n_runs - 1}\n")

    # 循环执行main.py
    for i in range(n_runs):
        # 当前轮次的seed（seed和split-seed保持一致，从42开始递增）
        current_seed = start_seed + i
        run_idx = i + 1  # 轮次编号（1~10）

        print(f"=== 第{run_idx:2d}次运行 | 当前seed/split-seed：{current_seed} ===")
        try:
            # 构造命令行参数：调用当前Python环境执行main.py，并传递seed参数
            cmd = [
                sys.executable,  # 确保使用当前环境的Python解释器（避免环境冲突）
                "main.py",  # 目标脚本（需与当前运行脚本在同一目录，或写绝对路径）
                "--seed", str(current_seed),
                "--split-seed", str(current_seed)  # split-seed与seed保持一致
            ]

            # 执行命令并捕获输出（stdout/stderr实时打印，便于调试）
            result = subprocess.run(
                cmd,
                check=True,  # 若main.py返回非0退出码，抛出异常
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # 输出按字符串处理（而非字节流）
                encoding="utf-8"  # 解决中文输出乱码问题
            )

            # 打印当前轮次的正常输出
            print(f"第{run_idx:2d}次运行成功！输出日志：")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # 捕获main.py执行失败的异常（如代码报错、参数错误等）
            print(f"❌ 第{run_idx:2d}次运行失败！错误信息：")
            print(f"返回码：{e.returncode}")
            print(f"错误输出：{e.stderr}")
            print(f"继续执行下一次运行...\n")
            continue  # 即使当前轮次失败，仍继续下一轮次
        except FileNotFoundError:
            print(f"❌ 第{run_idx:2d}次运行失败：未找到main.py！请确认脚本路径正确。")
            break  # 脚本不存在，直接终止循环
        except Exception as e:
            # 捕获其他意外异常（如权限问题、环境问题等）
            print(f"❌ 第{run_idx:2d}次运行遇到未知错误：{str(e)}")
            print(f"继续执行下一次运行...\n")
            continue
        print(f"第{run_idx:2d}次运行完成 | 耗时：{datetime.now() - start_time}\n")

    # 记录整体结束时间
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"=== 批量运行结束 | 结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"总运行次数：{n_runs}次 | 总耗时：{total_duration}")
    print(f"成功次数：{n_runs - sum(1 for i in range(n_runs) if '失败' in locals())}次（需结合上方日志确认）")


if __name__ == "__main__":
    # 调用函数：默认从seed=42开始，运行10次pi
    run_main_multiple_times(start_seed=42, n_runs=10)