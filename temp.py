import os


nci_dir = os.path.join(os.path.dirname(__file__), "data", "raw", "nci")

for nci_subd in os.listdir(nci_dir):
    complete_path = os.path.join(nci_dir, nci_subd)
    for md in os.listdir(complete_path):
        print( nci_subd + "/" + md)