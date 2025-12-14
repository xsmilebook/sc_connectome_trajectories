import os

folder_path = "D:\projects\sc_connectome_trajectories\data\ABCD\sc_connectome\schaefer400"

for file_name in os.listdir(folder_path):
    if file_name.endswith("invnode.csv"):
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")


