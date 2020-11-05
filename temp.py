import json


def find(file_path, key=None, value=None):
    data_key = dict()
    data_value = dict()
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i in data.items():
        print(i)
    return data_key, data_value


def main():
    a, b = find("shit.json")


if __name__ == '__main__':
    main()
