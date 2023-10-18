import json
from typing import Any, Union, List
from matplotlib import pyplot as plt

class record():
    def __init__(self, savePath:str) -> None:
        self.savePath = savePath
        self.log = []
        self.R = {}

    def set_record(self, name:str, value:Any):
        self.R.update({name: value})

    def step(self):
        self.log.append(self.R)
        self.R = {}

    def save(self):
        with open(self.savePath, "w", encoding="utf-8") as f:
            for R in self.log:
                f.write(json.dumps(R, ensure_ascii=False)+'\n')

    def clear(self):
        self.log = []

    def show_record(self, name:Union[str, List[str]]):
        if len(self.log) == 0:
            with open(self.savePath, "r", encoding="utf-8") as f:
                for R in f.readlines():
                    self.log.append(json.loads(R))

        showData = []
        for D in self.log:
            if isinstance(name, str):
                showData.append(D[name])
            elif isinstance(name, list):
                showData.append([D[N] for N in name])

        plt.plot(showData, label = name)
        plt.legend()
        plt.show()

def main():
    log_record = record("./log.json")
    log_record.show_record(["reward_max", "reward_min"])

if __name__ == "__main__":
    main() 