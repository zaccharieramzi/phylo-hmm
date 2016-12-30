import json


def save_tree(filePath, tree):
    with open(filePath, 'w') as f:
        json.dump(tree, f, sort_keys=True, indent=4,
                  separators=(',', ': '))
        print("Tree saved to {:s}".format(file_path))


def load_tree(filePath):
    tree = {}
    with open(filePath) as f:
        tree = json.load(f)
        # we want key as integers
        keys = list(tree.keys())
        for key in keys:
            if key.isdigit():
                tree[int(key)] = tree[key]
                tree.pop(key, None)
    return tree


if __name__ == '__main__':
    # nodes numeroted from 1 to 15
    tree = {
        15: [{"node": 13, "branch": 0.2}, {"node": 14, "branch": 0.1}],
        14: [{"node": 11, "branch": 0.2}, {"node": 12, "branch": 0.1}],
        13: [{"node": 9, "branch": 0.2}, {"node": 10, "branch": 0.1}],
        12: [{"node": 7, "branch": 0.2}, {"node": 8, "branch": 0.1}],
        11: [{"node": 5, "branch": 0.2}, {"node": 6, "branch": 0.1}],
        10: [{"node": 3, "branch": 0.2}, {"node": 4, "branch": 0.1}],
        9: [{"node": 1, "branch": 0.2}, {"node": 2, "branch": 0.1}]
    }
    for i in range(1, 9):
        tree[i] = []
    save_tree("tree.json", tree)
