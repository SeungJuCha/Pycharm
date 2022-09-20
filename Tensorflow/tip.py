import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""gpu가 있는 상황에서도 cpu 사용을 원한다면
아래와 같이 디바이스의 번호를 지정해주시면 됩니다.
CPU 강제 사용을 원하신다면 -1로 번호를 선택해주시면 됩니다.
혹은 with 구문을 통해서도 특정 부분의 코드에서만 디바이스를 지정할 수 있습니다."""
# GPU 사용을 원하는 경우
with tf.device('/device:GPU:0'):
    # 원하는 코드 작성(들여쓰기 필수)

# CPU 사용을 원하는 경우
with tf.device('/cpu:0'):
# 원하는 코드 작성(들여쓰기 필수
import tensorflow as tf
tf.__version__                    # 텐서플로 버전확인
from tensorflow.python.client import device_lib
device_lib.list_local_devices()   # GPU를 사용하는 지 확인