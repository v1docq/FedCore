import numpy as np
import matplotlib.pyplot as plt


class Individual(object):
    def __init__(self, dv_list: list, obj_list):
        self.dv = np.array(dv_list)
        self.obj = np.array(obj_list)
        self.n_dv = len(self.dv)
        self.n_obj = len(self.obj)

    # selfがotherを支配する場合 -> True
    def dominate(self, other) -> bool:
        if not isinstance(other, Individual):
            Exception("not indiv.")

        if all(s <= o for s, o in zip(self.obj, other.obj)) and \
                any(s != o for s, o in zip(self.obj, other.obj)):
            return True
        return False


def indiv_sort(population, key=-1):
    popsize = len(population)
    if popsize <= 1:
        return population

    pivot = population[0]
    left = []
    right = []
    for i in range(1, popsize):
        indiv = population[i]
        # print(indiv.obj, indiv.obj[key], pivot.obj[key])
        if indiv.obj[key] <= pivot.obj[key]:
            left.append(indiv)
        else:
            right.append(indiv)
    # print()
    left = indiv_sort(left, key)
    right = indiv_sort(right, key)

    center = [pivot]
    return left + center + right


class NonDominatedSort(object):

    def __init__(self):
        pass
        # self.pop = pop

    def sort(self, population: list, return_rank=False):
        popsize = len(population)

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        for i in range(popsize):
            for j in range(popsize):
                # if i == j:
                #     continue
                # iがjに優越されている -> True
                dom = population[j].dominate(population[i])
                is_dominated[i, j] = (i != j) and dom

        # iを優越する個体の数
        is_dominated.sum(axis=(1,), out=num_dominated)
        # print(num_dominated)

        fronts = []
        limit = popsize
        for r in range(popsize):
            front = []
            for i in range(popsize):
                is_rank_ditermined = not (rank[i] or num_dominated[i])
                mask[i] = is_rank_ditermined
                if is_rank_ditermined:
                    rank[i] = r + 1
                    front.append(population[i])

            fronts.append(front)
            limit -= len(front)

            if return_rank:
                if rank.all():
                    return rank
            elif limit <= 0:
                return fronts

            # print(np.sum(mask & is_dominated))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception("Error: reached the end of function")


class HyperVolume(object):
    def __init__(self, pareto, ref_points: list):
        self.pareto = pareto
        self.pareto_sorted = indiv_sort(self.pareto)
        self.ref_point = np.ones(pareto[0].n_obj)
        self.ref_point = np.array(ref_points)

        self.obj_dim = pareto[0].n_obj
        self.volume = 0

        self.calcpoints = []

    def set_refpoint(self, opt="minimize"):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)
        pareto_arr = np.array(pareto_arr)

        minmax = max
        if opt == "maximize":
            minmax = min

        for i in range(len(self.ref_point)):
            self.ref_point[i] = minmax(pareto_arr[:, i])

    def hso(self):
        pl, s = self.obj_dim_sort(self.pareto)
        s = [pl, s]
        for k in range(self.obj_dim):
            s_dash = []

    def calculate(self, obj_dim):
        pass

    def calc_2d(self):
        if len(self.ref_point) != 2:
            return NotImplemented

        vol = 0
        b_indiv = None

        for i, indiv in enumerate(self.pareto_sorted):
            if i == 0:
                x = (self.ref_point[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])
            else:
                x = (b_indiv.obj[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])

            self.calcpoints.append([x, y])
            vol += x * y
            b_indiv = indiv
            # print(f"vol:{vol:.10f}  x:{x:.5f}  y:{y:.5f}")

        self.volume = vol
        self.calcpoints = np.array(self.calcpoints)
        return vol

    def obj_dim_sort(self, dim=-1):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)

        pareto_arr = np.array(pareto_arr)
        res_arr = pareto_arr[pareto_arr[:, dim].argsort(), :]
        self.pareto_sorted = res_arr

        return res_arr, res_arr[:, dim]


def indiv_plot(population: list, color=None):
    evals = []
    for indiv in (population):
        # print(indiv)
        evals.append(indiv.obj)

    evals = np.array(evals)

    plt.scatter(evals[:, 0], evals[:, 1], c=color)


def data_save(pareto, vol, ref_point, fname, ext="txt"):
    pareto_arr = []
    for indiv in pareto:
        pareto_arr.append(indiv.obj)
    pareto_arr = np.array(pareto_arr)

    delimiter = " "
    if ext == "csv":
        delimiter = ","

    np.savetxt(fname + "_pareto." + ext, pareto_arr, delimiter=delimiter)
    with open(fname + "_HV." + ext, "w") as f:
        f.write("#HyperVolume\n")
        f.write(f"{vol}\n")

        f.write("#ref_point\n")
        for p in ref_point:
            f.write(f"{p}, ")
        f.write("\n")


def main():
    input_fname = "tablex.txt"  # input file name
    output_fname = "result_data"  # result file name
    ext = "txt"  # outputファイルの拡張子
    ref_points = [1.0, 1.0]

    # front = np.array([[11,4,4],
    #                   [9,2,5],
    #                   [5,6,7],
    #                   [3,3,10]])

    # front = np.array([[11,4],
    #                   [9,2],
    #                   [5,6],
    #                   [3,3]])

    # データの取得 & non-dominated-sort
    dat = np.loadtxt(input_fname, skiprows=1)
    sortfunc = NonDominatedSort()
    population = []
    for s in dat:
        population.append(Individual(s[1:6], s[6:]))
        # print(population[-1].__dict__)

    front = sortfunc.sort(population)

    pareto = front[0]  # パレート解のリスト
    print("Number of pareto solutions: ", len(pareto))

    # calculate HV
    hypervol = HyperVolume(pareto, [1.0, 1.0])
    vol = hypervol.calc_2d()
    print("ref_point: ", hypervol.ref_point)
    print("HV: ", vol)

    # HVなどの出力
    data_save(hypervol.pareto_sorted, vol, hypervol.ref_point, output_fname, ext=ext)

    # plot all indiv(blue) and pareto indiv(red)
    indiv_plot(population)
    indiv_plot(pareto, color="Red")
    # plt.scatter(hypervol.calcpoints[:,0], hypervol.calcpoints[:,1], "*")
    print(hypervol.calcpoints)
    plt.show()


if __name__ == "__main__":
    main()