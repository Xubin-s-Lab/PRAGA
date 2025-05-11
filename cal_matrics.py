import argparse
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score
import os


def read_list_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}\nPlease confirm that the main program has successfully generated this result file.")
    lst = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            lst.append(num)
    return lst

def main(arg):
    GT_list = read_list_from_file(args.GT_path)
    # SG_list = read_list_from_file(args.SpatialGlue_path)
    Our_list = read_list_from_file(args.our_path)

    # Our_list = [x+1 for x in Our_list]

    # label_mapping1 = {k: real for k, real in zip(range(0, len(set(GT_list))), set(GT_list))}
    # Our_list = [label_mapping1[label] for label in Our_list]

    print(min(GT_list), max(GT_list))
    print(min(Our_list), max(Our_list))
    print(set(GT_list))
    print(len(GT_list))
    print(len(Our_list))

    Our_Jaccard = jaccard(Our_list, GT_list)
    print(f"our         jaccard: {Our_Jaccard:.6f}")


    Our_F_measure = F_measure(Our_list, GT_list)
    print(f"our         F_measure: {Our_F_measure:.6f}")


    Our_mutual_info = mutual_info_score(GT_list, Our_list)
    print(f"our         Mutual Information: {Our_mutual_info:.6f}")

    Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
    print(f"Our         (NMI): {Our_nmi:.6f}")

    Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
    print(f"Our         (AMI): {Our_ami:.6f}")

    Our_V = v_measure_score(GT_list, Our_list)
    print(f"Our         V-measure: {Our_V:.6f}")

    Our_homogeneity = homogeneity_score(GT_list, Our_list)
    Our_completeness = completeness_score(GT_list, Our_list)
    print(f"Our         Homogeneity: {Our_homogeneity:.6f} Completeness: {Our_completeness:.6f}")

    Our_ari = adjusted_rand_score(GT_list, Our_list)
    print(f"Our         (ARI): {Our_ari:.6f}")

    Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
    print(f"Our         (FMI): {Our_fmi:.6f}")

    with open(arg.save_path, 'w') as f:
        f.write(f"Our     jaccard: {Our_Jaccard:.6f}\n")
        f.write(f"Our     F_measure: {Our_F_measure:.6f}\n")
        f.write(f"Our     Mutual Information: {Our_mutual_info:.6f}\n")
        f.write(f"Our     NMI: {Our_nmi:.6f}\n")
        f.write(f"Our     AMI: {Our_ami:.6f}\n")
        f.write(f"Our     V-measure: {Our_V:.6f}\n")
        f.write(f"Our     Homogeneity: {Our_homogeneity:.6f}\n")
        f.write(f"Our     Completeness: {Our_completeness:.6f}\n")
        f.write(f"Our     (ARI): {Our_ari:.6f}\n")
        f.write(f"Our     (FMI): {Our_fmi:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--GT_path', default='./HLN/GT_labels.txt', type=str, help='GT_path')
    # parser.add_argument('--SpatialGlue_path', default='', type=str, help='SpatialGlue_path')
    parser.add_argument('--our_path', default='./results/HLN.txt', type=str, help='our_path')
    parser.add_argument('--save_path', default='./results/HLN_metrics.txt', type=str, help='our_path')

    args = parser.parse_args()
    main(args)
