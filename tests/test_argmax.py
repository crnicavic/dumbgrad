from dumbgrad.nn import argmax
import random

if __name__ == "__main__":
    arr_size = 100
    arr = [0] * arr_size
    for i, a in enumerate(arr):
        arr[i] = 10
        if argmax(arr) != i:
            print(f"argmax broken!\n, max is at {i}, but got {argmax(arr)}")
            break
        arr[i] = 0

    test_count = 100
    test_arr_size = 100
    arr = [random.random() for _ in range(test_arr_size)]
    for _ in range(test_count):
        random.shuffle(arr)
        if arr[argmax(arr)] != max(arr):
            print(f"argmax broken!\n, max is at {i}, but got {argmax(arr)}")
            break
