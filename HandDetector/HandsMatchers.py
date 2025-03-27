from abc import ABC, abstractmethod
import numpy as np


class HandsMatcher(ABC):
    @abstractmethod
    def __init__(self, matched_ls: list[str]):
        self.matched_ls: list[str] = matched_ls
        self.n = len(matched_ls)
        self.row_n: int = 0  # 记录当前检测到的行/数量
        self.matched_dict: dict[int, str] = dict()

    @abstractmethod
    def run(self, cost_matrix: np.ndarray) -> dict:
        return self.matched_dict


class HungarianMatcher(HandsMatcher):
    def __init__(self, matched_ls: list[str]):
        """
        Args:
            matched_ls: 需要被匹配的手部名字列表
        """
        self.matched_ls: list[str] = matched_ls
        self.n: int = len(matched_ls)
        self.row_n: int = 0  # 记录当前检测到的行/数量
        self.matched_dict: dict[int, str] = dict()

    def _extend2square(self, cost_matrix: np.ndarray) -> np.ndarray:
        """用0行填充剩下没有检测到的手,扩展成方阵
        Args:
            cost_matrix: 未扩展的初始的成本矩阵,列数要和被检测手部数量相同
        """
        self.row_n, col_n = cost_matrix.shape
        if self.row_n < self.n:  # 添加没有检测到的行(虚拟行),扩展成方阵才能用该算法
            return np.vstack([cost_matrix, np.zeros((self.n - self.row_n, col_n))])
        return cost_matrix.copy()

    def _reduce_matrix(self, cur_matrix: np.ndarray) -> None:
        """矩阵归约,生成尽可能多的零元素
        Args:
            cur_matrix: 当前处理过之后的方阵
        """
        # 行归约:每行减去最小值,得到零元素
        row_mins = cur_matrix.min(axis=1, keepdims=True)
        cur_matrix -= row_mins
        # 列归约:每列减去最小值
        col_mins = cur_matrix.min(axis=0, keepdims=True)
        cur_matrix -= col_mins

    def _find_0line(self, cur_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """找到最优的零覆盖线,使用贪心策略
        Args:
            cur_matrix: 当前处理过之后的方阵
        """
        # assert cost_matrix.shape[0] == n and cost_matrix.shape[1] == n
        # 获取零元素所在位置的方阵
        zeros_matrix = cur_matrix == 0  # 得到布尔值矩阵,零元素位置为True
        # 定义划线位置,row_line记录横着画的线,col_line记录竖线
        row_line = np.zeros(self.n, dtype=np.bool)
        col_line = np.zeros(self.n, dtype=np.bool)
        while np.sum(zeros_matrix) > 0:  # 不断画线直到覆盖到全部的零元素
            # 在最多零的地方画横线
            row_line[np.argmax(np.sum(zeros_matrix, axis=1))] = True
            zeros_matrix[row_line, :] = False  # 将已经画了线的地方设为False
            if np.sum(zeros_matrix) <= 0:
                break  # 每次画完一条线都检测是否完全覆盖,横线竖线交替画
            col_line[np.argmax(np.sum(zeros_matrix, axis=0))] = True  # 画竖线
            zeros_matrix[:, col_line] = False
        return row_line, col_line

    def _adjust_matrix(
        self, cur_matrix: np.ndarray, row_line: np.ndarray, col_line: np.ndarray
    ) -> None:
        """调整矩阵,在未覆盖区域生成新的零,同时保持已有零的位置不变
        Args:
            cur_matrix: 当前处理过之后的方阵
            row_line: 行的画线标志数组
            col_line: 列的画线标志数组
        """
        # 标记覆盖区域
        covered_matrix = np.zeros_like(cur_matrix, dtype=np.bool)
        covered_matrix[row_line, :] = True  # 标记横线覆盖区域
        covered_matrix[:, col_line] = True  # 标记竖线覆盖区域
        # 获取未覆盖区域的最小值
        uncovered_min = np.min(cur_matrix[~covered_matrix])
        # 对未覆盖区域进行减法,在未覆盖区域生成新零
        cur_matrix[~covered_matrix] -= uncovered_min
        # 对覆盖线交叉的位置进行加法,防止已有零被破坏
        cur_matrix[row_line, col_line] += uncovered_min

    def _mark_in_zeros(self, cur_matrix: np.ndarray) -> dict[int, str]:
        """标记独立零,通过找独立零来匹配两边的节点
        Args:
            cur_matrix: 当前处理过之后的方阵
        """
        # 获取零元素所在位置的方阵
        zeros_matrix = cur_matrix == 0  # 得到布尔值矩阵,零元素位置为True
        for _ in range(self.row_n):
            # 贪心策略获取独立零
            zero_num = np.sum(zeros_matrix, axis=1)  # 统计每行零的数量
            zero_num[zero_num <= 0] = self.n  # 将已经选过的行设为最大(减少取到该行)
            zero_num[self.row_n + 1 :] = self.n  # 将没有检测到的行设为最大
            row_idx = int(np.argmin(zero_num))  # 取零数量最少的行作为本行的独立零
            col_idx = int(np.argmax(zeros_matrix[row_idx, :]))  # 获取对应的列索引
            # 清空所在行和列的0
            zeros_matrix[row_idx, :] = False
            zeros_matrix[:, col_idx] = False
            # 对应匹配,检查是否匹配到没有检测到的行
            if row_idx < self.row_n:
                self.matched_dict[row_idx] = self.matched_ls[col_idx]
        return self.matched_dict

    def run(self, cost_matrix: np.ndarray) -> dict[int, str]:
        """运行匈牙利算法,返回匹配后的字典
        Args:
            cost_matrix: 未扩展的初始的成本矩阵,列数要和被检测手部数量相同
        """
        # assert cost_matrix.ndim == 2 and cost_matrix.shape[1] == self.n
        if cost_matrix.shape[1] != self.n:
            return {}
        self.matched_dict.clear()  # 清空上次的匹配结果
        # 扩展为方阵,才能使用匈牙利算法
        matrix = self._extend2square(cost_matrix)
        self._reduce_matrix(matrix)
        row_line, col_line = self._find_0line(matrix)
        # 用覆盖线数量来判定是否完美匹配
        while (row_line.sum() + col_line.sum()) < self.n:
            self._adjust_matrix(matrix, row_line, col_line)  # 调整矩阵
            row_line, col_line = self._find_0line(matrix)  # 更新覆盖线
        return self._mark_in_zeros(matrix)


if __name__ == "__main__":
    # 测试
    matched_ls = ["0", "1", "2"]
    matcher = HungarianMatcher(matched_ls)  # 创建匈牙利算法实例

    cost_matrix = np.array([[3, 1, 2], [2, 4, 7], [3, 6, 8]])
    matches = matcher.run(cost_matrix)
    print(1, matches)  # res: 0->2, 1->1, 2->0

    cost_matrix = np.arange(9).reshape(3, 3)
    matches = matcher.run(cost_matrix)
    print(2, matches)  # res: 0->0, 1->1

    cost_matrix = np.array([[3, 1, 2], [2, 4, 7]])
    matches = matcher.run(cost_matrix)
    print(3, matches)  # res: 0->1, 1->0

    cost_matrix = np.array([[2, 2, 3], [2, 2, 3], [3, 3, 4]])
    matches = matcher.run(cost_matrix)
    print(4, matches)  # res: 0->0, 1->1, 2->2 或 0->1, 1->0, 2->2

    cost_matrix = np.array([[3, -1, 2], [2, 4, 7], [-3, 6, 8]])
    matches = matcher.run(cost_matrix)
    print(5, matches)  # res: 0->1, 1->2, 2->0

    cost_matrix = np.zeros((3, 3))
    matches = matcher.run(cost_matrix)
    print(6, matches)  # res: 0->0, 1->1, 2->2等任意匹配都可以

    matched_ls = ["0", "1", "2", "3"]
    matcher = HungarianMatcher(matched_ls)  # 创建匈牙利算法实例
    cost_matrix = np.array([[5, 3, 6, 2], [4, 7, 1, 3], [2, 5, 4, 8], [9, 2, 7, 5]])
    matches = matcher.run(cost_matrix)
    print(7, matches)  # res: 0->3, 1->2, 2->0, 3->1
