import yaml

with open('test.yaml', 'r') as f:
    doc = yaml.load(f, Loader=yaml.FullLoader)
    print(doc)