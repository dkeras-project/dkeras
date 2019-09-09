import time
import ray


@ray.remote(num_cpus=0.1)
def test():
    time.sleep(10)
    return 1


def main():
    ray.init()

    workers = [test.remote() for _ in range(40)]
    print(ray.get(workers))


if __name__ == "__main__":
    main()
