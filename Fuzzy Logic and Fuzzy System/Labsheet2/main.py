import numpy as np
from matplotlib import pyplot as plt


def left_near_mf(x):
    print(type(x))
    if 0 <= x <= 15:
        return 1
    elif 15 <= x <= 30:
        return (30 - x) / 15
    elif 30 <= x <= 50:
        return 0
    else:
        return 0

def left_far_mf(x):
    if 0 <= x <= 20:
        return 0
    elif 20 <= x <= 35:
        return (x - 20) / 15
    elif 35 <= x <= 50:
        return 1

# def left_ok_mf(x):
#     if 0 <= x <= 25:
#         return x / 25
#     elif 25 <= x <= 50:
#         return (50 - x) / 25

def right_near_mf(y):
    if 0 <= y <= 15:
        return 1
    elif 15 <= y <= 30:
        return (30 - y) / 15
    elif 30 <= y <= 50:
        return 0

def right_far_mf(y):
    if 0 <= y <= 20:
        return 0
    elif 20 <= y <= 35:
        return (y - 20) / 15
    elif 35 <= y <= 50:
        return 1

# def right_ok_mf(y):
#     if 0 <= y <= 25:
#         return y / 25
#     elif 25 <= y <= 50:
#         return (50 - y) / 25

def output_left_mf(z):
    if 0 <= z <= 40:
        return (40 - z) / 40
    # elif 10 <= z <= 40:
    #     return (40 - z) / 40
    # elif 40 <= z <= 60:
    #     return 0
    # elif 60 <= z <= 90:
    #     return 0
    elif 40 <= z <= 100:
        return 0

def output_right_mf(z):
    if 0 <= z <= 60:
        return 0
    elif 60 <= z <= 100:
        return (z - 60) / 40

def output_straight_mf(z):
    if 0 <= z <= 10:
        return 0
    elif 10 <= z <= 50:
        return (z - 10) / 40
    elif 50 <= z <= 90:
        return (90 - z) / 40
    elif 90 <= z <= 100:
        return 0

def fuzzy_left_set(distance_left):
    left_near = left_near_mf(distance_left)
    # left_ok = left_ok_mf(distance_left)
    left_far = left_far_mf(distance_left)
    return left_far, left_near

def fuzzy_right_set(distance_right):
    right_near = right_near_mf(distance_right)
    # right_ok = right_ok_mf(distance_right)
    right_far = right_far_mf(distance_right)
    return right_near, right_far

def plot_left_mf(distance_left):
    x = range(0, 51)
    q_near = [left_near_mf(i) for i in x]
    # q_ok = [left_ok_mf(i) for i in x]
    q_far = [left_far_mf(i) for i in x]

    plt.plot(x, q_near, label='Near')
    # plt.plot(x, q_ok, label='Ok')
    plt.plot(x, q_far, label='Far')
    plt.axvline(x=distance_left, color='gray', linestyle='--')
    plt.legend()
    plt.show()

def plot_right_mf(distance_right):
    y = range(0, 51)
    p_near = [right_near_mf(i) for i in y]
    # p_ok = [right_ok_mf(i) for i in y]
    p_far = [right_far_mf(i) for i in y]

    plt.plot(y, p_near, label='Near')
    # plt.plot(y, p_ok, label='Ok')
    plt.plot(y, p_far, label='Far')
    plt.axvline(x=distance_right, color='gray', linestyle='--')
    plt.legend()
    plt.show()

def plot_output_mf():
    z = range(0, 101)
    r_left = [output_left_mf(i) for i in z]
    r_straight = [output_straight_mf(i) for i in z]
    r_right = [output_right_mf(i) for i in z]

    plt.plot(z, r_left, label='Left')
    plt.plot(z, r_straight, label='Straight')
    plt.plot(z, r_right, label='Right')
    # plt.axvline(x=direction, color='gray', linestyle='--')
    plt.legend()
    plt.show()

# def defuzzication_mf(m):
#     if 0 <= m <= 10:
#         return min(output_left_mf(m), min(left_far_mf(m), right_near_mf(m)))
#     elif 10 <= m <= 40:
#         return np.max([min(output_left_mf(m), min(left_far_mf(m), right_near_mf(m))), min(output_straight_mf(m), min(left_far_mf(m), right_far_mf(m)))])
#     elif 40 <= m <= 60:
#         return np.min([output_straight_mf(m), min(left_far_mf(m), right_far_mf(m))])
#     elif 60 <= m <= 90:
#         return np.max([min(output_straight_mf(m), min(left_far_mf(m), right_far_mf(m))), min(output_right_mf(m), max(min(left_near_mf(m), right_near_mf(m)), min(left_near_mf(m), right_far_mf(m))))])
#     elif 90 <= m <= 100:
#         return np.min([output_right_mf(m), max(min(left_near_mf(m), right_near_mf(m)), min(left_near_mf(m), right_far_mf(m)))])

def main():
    # distance_left = int(input("Enter left distance in cm: "))
    # distance_right = int(input("Enter right distance in cm: "))
    # left_far, left_near = fuzzy_left_set(distance_left)
    # right_near, right_far = fuzzy_right_set(distance_right)
    # u1 = min(left_near, right_near)
    # u2 = min(left_near, right_far)
    # u3 = min(left_far, right_near)
    # u4 = min(left_far, right_far)
    # u = max(u1, u2)
    # x_cen_up = np.array([0 * min(output_left_mf(0), u3), 5 * min(output_left_mf(5), u3), 10 * min(output_left_mf(10), u3),
    #                     15 * max(min(output_left_mf(15), u3), min(output_straight_mf(15), u4)),
    #                     20 * max(min(output_left_mf(20), u3), min(output_straight_mf(20), u4)),
    #                     25 * max(min(output_left_mf(25), u3), min(output_straight_mf(25), u4)),
    #                     30 * max(min(output_left_mf(30), u3), min(output_straight_mf(30), u4)),
    #                     35 * max(min(output_left_mf(35), u3), min(output_straight_mf(35), u4)),
    #                     40 * min(output_straight_mf(40), u4), 45 * min(output_straight_mf(45), u4),
    #                     50 * min(output_straight_mf(50), u4), 55 * min(output_straight_mf(55), u4),
    #                     60 * min(output_straight_mf(60), u4),
    #                     65 * max(min(output_straight_mf(65), u4), min(output_right_mf(65), u)),
    #                     70 * max(min(output_straight_mf(70), u4), min(output_right_mf(70), u)),
    #                     75 * max(min(output_straight_mf(75), u4), min(output_right_mf(75), u)),
    #                     80 * max(min(output_straight_mf(80), u4), min(output_right_mf(80), u)),
    #                     85 * max(min(output_straight_mf(85), u4), min(output_right_mf(85), u)),
    #                     90 * min(output_right_mf(90), u), 95 * min(output_right_mf(95), u), 100 * min(output_right_mf(100), u)])
    # x_cen_down = np.array([min(output_left_mf(0), u3), min(output_left_mf(5), u3), min(output_left_mf(10), u3),
    #                       max(min(output_left_mf(15), u3), min(output_straight_mf(15), u4)),
    #                       max(min(output_left_mf(20), u3), min(output_straight_mf(20), u4)),
    #                       max(min(output_left_mf(25), u3), min(output_straight_mf(25), u4)),
    #                       max(min(output_left_mf(30), u3), min(output_straight_mf(30), u4)),
    #                       max(min(output_left_mf(35), u3), min(output_straight_mf(35), u4)),
    #                       min(output_straight_mf(40), u4), min(output_straight_mf(45), u4),
    #                       min(output_straight_mf(50), u4), min(output_straight_mf(55), u4),
    #                       min(output_straight_mf(60), u4),
    #                       max(min(output_straight_mf(65), u4), min(output_right_mf(65), u)),
    #                       max(min(output_straight_mf(70), u4), min(output_right_mf(70), u)),
    #                       max(min(output_straight_mf(75), u4), min(output_right_mf(75), u)),
    #                       max(min(output_straight_mf(80), u4), min(output_right_mf(80), u)),
    #                       max(min(output_straight_mf(85), u4), min(output_right_mf(85), u)),
    #                       min(output_right_mf(90), u), min(output_right_mf(95), u), min(output_right_mf(100), u)])
    # x_cen = x_cen_up.sum() / x_cen_down.sum()
    # print("left_far:", left_far)
    # print("left_near:", left_near)
    # print("right_far:", right_far)
    # print("right_near:", right_near)
    # print("centroid defuzzification:", x_cen)
    # plot_right_mf(distance_right)
    # plot_left_mf(distance_left)
    # plot_output_mf()
    #
    # m = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    # plt.figure('Defuzzification')
    # ax = plt.gca()
    # ax.plot(m, x_cen_down, color='r', linewidth=1, alpha=0.6)
    #
    # plt.show()

    # ax3 = plt.axes(projection='3d')

    # 定义三维数据
    distance_left = np.arange(0, 50)
    distance_right = np.arange(0, 50)

    result_surface = []

    for n1 in distance_left:
        for n2 in distance_right:
            left_far, left_near = fuzzy_left_set(distance_left)
            right_near, right_far = fuzzy_right_set(distance_right)
            u1 = min(left_near, right_near)
            u2 = min(left_near, right_far)
            u3 = min(left_far, right_near)
            u4 = min(left_far, right_far)
            u = max(u1, u2)
            x_cen_up = np.array([0 * min(output_left_mf(0), u3), 5 * min(output_left_mf(5), u3), 10 * min(output_left_mf(10), u3),
                                15 * max(min(output_left_mf(15), u3), min(output_straight_mf(15), u4)),
                                20 * max(min(output_left_mf(20), u3), min(output_straight_mf(20), u4)),
                                25 * max(min(output_left_mf(25), u3), min(output_straight_mf(25), u4)),
                                30 * max(min(output_left_mf(30), u3), min(output_straight_mf(30), u4)),
                                35 * max(min(output_left_mf(35), u3), min(output_straight_mf(35), u4)),
                                40 * min(output_straight_mf(40), u4), 45 * min(output_straight_mf(45), u4),
                                50 * min(output_straight_mf(50), u4), 55 * min(output_straight_mf(55), u4),
                                60 * min(output_straight_mf(60), u4),
                                65 * max(min(output_straight_mf(65), u4), min(output_right_mf(65), u)),
                                70 * max(min(output_straight_mf(70), u4), min(output_right_mf(70), u)),
                                75 * max(min(output_straight_mf(75), u4), min(output_right_mf(75), u)),
                                80 * max(min(output_straight_mf(80), u4), min(output_right_mf(80), u)),
                                85 * max(min(output_straight_mf(85), u4), min(output_right_mf(85), u)),
                                90 * min(output_right_mf(90), u), 95 * min(output_right_mf(95), u), 100 * min(output_right_mf(100), u)])
            x_cen_down = np.array([min(output_left_mf(0), u3), min(output_left_mf(5), u3), min(output_left_mf(10), u3),
                                  max(min(output_left_mf(15), u3), min(output_straight_mf(15), u4)),
                                  max(min(output_left_mf(20), u3), min(output_straight_mf(20), u4)),
                                  max(min(output_left_mf(25), u3), min(output_straight_mf(25), u4)),
                                  max(min(output_left_mf(30), u3), min(output_straight_mf(30), u4)),
                                  max(min(output_left_mf(35), u3), min(output_straight_mf(35), u4)),
                                  min(output_straight_mf(40), u4), min(output_straight_mf(45), u4),
                                  min(output_straight_mf(50), u4), min(output_straight_mf(55), u4),
                                  min(output_straight_mf(60), u4),
                                  max(min(output_straight_mf(65), u4), min(output_right_mf(65), u)),
                                  max(min(output_straight_mf(70), u4), min(output_right_mf(70), u)),
                                  max(min(output_straight_mf(75), u4), min(output_right_mf(75), u)),
                                  max(min(output_straight_mf(80), u4), min(output_right_mf(80), u)),
                                  max(min(output_straight_mf(85), u4), min(output_right_mf(85), u)),
                                  min(output_right_mf(90), u), min(output_right_mf(95), u), min(output_right_mf(100), u)])
            x_cen = x_cen_up.sum() / x_cen_down.sum()
            result_surface.append(n1, n2, x_cen)
    result_x = [distance_left[0] for distance_left in result_surface]
    print(result_surface)
    print(result_x)

if __name__ == "__main__":
    main()

# result = []
# x = range(1,10)
# y = range(1,10)
# for i in x:
#     for j in y:
#         z = i*j
#         result.append((i,j,z))
# res = [x[0] for x in result]
# print(result)
# print(res)
#
# for i in range(distance_left.shape[1]):
#     for j in range(distance_right.shape[1]):
#         ax[i, j] = x_cen(distance_left[i], distance_right[j])
# ax3.plot_surface(distance_left, distance_right, ax, cmap='rainbow')
# plt.show()