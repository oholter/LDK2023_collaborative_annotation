from utils.io import read_flat_file, get_order_from_file
import logging
import itertools
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    outliers = []
    for x in data:
        z_score = (x - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(x)
    data = [x for x in data if x not in outliers]
    return data


def gety(data, attribute, order=None):
    if not order:
        return np.array([float(value[attribute]) for key, value in data.items() \
                        if attribute in value])
    else:
        ys = []
        for idx in order:
            if idx in data and attribute in data[idx]:
                #print(data[idx])
                y = float(data[idx][attribute])
                ys.append(y)
        return np.array(ys)

def normalize_data(ys):
    min_y = min(ys)
    max_y = max(ys)
    normalized_ys = []
    for y in ys:
        normalized_y = (y - min_y) / (max_y - min_y)
        normalized_ys.append(normalized_y)

    return np.array(normalized_ys)


def create_aggregated_ys(ys):
    agg_ys = []
    sum = 0
    for y in ys:
        sum += y
        agg_ys.append(sum)
    return np.array(agg_ys)


def get_order_from_file(path):
    with path.open(mode='r') as F:
        order = []
        for line in F:
            order.append(int(line.strip()))
        print(order)
    return order


def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("input",
                        help='file with data, separate with \",\" for multiple files')
    parser.add_argument("key", help='key to create figure from')
    parser.add_argument("--save", "-s", help="path to save figure")
    parser.add_argument("--order", help="text file with order, separate with \",\" for multiple files, if used, it MUST contain one order file for each input file")
    parser.add_argument("--remove_outliers", "-r", help="remove outliers",
                        action="store_true")
    parser.add_argument("--normalize_data", "-n", help="normalize data to 0,1",
                        action="store_true")
    parser.add_argument("--aggregate", "-a", help="aggregate data",
                        action="store_true")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    args = parser.parse_args()

    #input_path = Path(args.input)
    inputs = args.input.split(",")
    input_paths = [Path(i) for i in inputs]




    for input_path in input_paths:
        if not input_path.exists():
            print("Cannot find file: {}, exiting...".format(input_path))
            exit()

    if args.order:
        order_paths = [Path(o) for o in args.order.split(",")]
        if len(order_paths) != len(input_paths):
            print("The number of input files is not equal to the number of order files")
            print("Exiting")
            exit()
        for order_path in order_paths:
            if not order_path.exists():
                print("Cannot find file: {}, exiting...".format(input_path))
                exit()

        orders = [get_order_from_file(o) for o in order_paths]



    datas = [read_flat_file(i) for i in input_paths]

    # preprocessing of data
    if args.order:
        ys = [gety(data, args.key, order) for data, order in zip(datas, orders)]
    else:
        ys = [gety(data, args.key) for data in datas]
    # y = gety(data, args.key, order)
    if args.remove_outliers:
        ys = np.array([remove_outliers(y) for y in ys])
    if args.normalize_data:
        ys = [normalize_data(y) for y in ys]
    if args.aggregate:
        ys = [create_aggregated_ys(y) for y in ys]

    print("ys: {}".format(ys))
    xs = [np.array(range(len(y))) for y in ys]
    #xs = [range(len(y)) for y in ys]
    #xs = [list(range(len(y))) for y in ys]
    print("xs: {}".format(xs))


    # polynomial trend line
    #poly_features = PolynomialFeatures(degree=8)
    #x_poly = poly_features.fit_transform(np.array(x).reshape(-1, 1))
    #model = LinearRegression()
    #model.fit(x_poly, y)
    #y_poly_pred = model.predict(x_poly)

    marker = itertools.cycle(('+', '.', 'o', '*'))
    fig, ax = plt.subplots()
    if args.aggregate:
        coefficients = [np.polyfit(x, y, 1) for x, y in zip(xs, ys)]
        y_estimates = [a * x + b for (a, b), x in zip(coefficients, xs)]
        #y_errors = [x.std() * np.sqrt(1/len(x)) + (x - x.mean())**2 / np.sum((x - x.mean())**2) for x in xs]
        residuals = [y - y_est for y, y_est in zip(ys, y_estimates)]
        stds = [np.std(residual) for residual in residuals]
        #print(residuals)
        #print(y_errors)
        #exit()
        #y_2errors = [2*y for y in y_errors]
        #polynomials = [np.poly1d(coeff) for coeff in coefficients]
        #y_trends = [poly(x) for poly, x in zip(polynomials, xs)]

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            y_est = y_estimates[i]
            std = stds[i]
            #y_error = y_2errors[i]
            #ax.plot(x, y, marker=next(marker))
            ax.plot(x, y)
            #ax.plot(x, y_est)
            ax.fill_between(x, y_est - std, y_est + std, alpha=0.2)


    else:
        for y, x in zip(ys, xs):
            ax.scatter(x,y)
            #plt.plot(x, y_poly_pred, color='green')

        coefficients = [np.polyfit(x, y, 1) for x, y in zip(xs, ys)]
        #y_estimates = [a * x + b for (a, b), x in zip(coefficients, xs)]
        #y_errors = [x.std() * np.sqrt(1/len(x)) + (x - x.mean())**2 / np.sum((x - x.mean())**2) for x in xs]
        #y_errors = [y.std() * np.sqrt(1/len(y)) + (y - y.mean())**2 / np.sum((y - y.mean())**2) for y in ys]
        y_errors = [y.std() for y in ys]
        #print(y_errors)
        #exit()
        polynomials = [np.poly1d(coeff) for coeff in coefficients]
        y_trends = [poly(x) for poly, x in zip(polynomials, xs)]
        residuals = [y - y_trend for y, y_trend in zip(ys, y_trends)]
        #print(residuals)
        stds = [np.std(residual) for residual in residuals]
        #print(stds)
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            std = stds[i]
            #y_estimate = y_estimates[i]
            y_trend = y_trends[i]
            y_error = y_errors[i]
            plt.plot(x, y_trend)
            #ax.fill_between(x, y_trend - y_error, y_trend + y_error, alpha=0.2)
            ax.fill_between(x, y_trend - std, y_trend + std, alpha=0.2)


    plt.xlim(0,136)
    plt.ylim(0)
    if args.xlabel:
        plt.xlabel(args.xlabel, fontsize=14)
    if args.ylabel:
        plt.ylabel(args.ylabel, fontsize=14)


    if args.save:
        plt.savefig(args.save)
        logging.info("saved figure to {}".format(args.save))
    else:
        plt.show()





if __name__ == '__main__':
    main()
