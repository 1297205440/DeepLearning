import atexit
import subprocess

def run_next_program():
    subprocess.run(['python', 'train.py'])  # 第二个函数是下一个程序的位置

if __name__ == '__main__':
    print(True)
    atexit.register(run_next_program)