import math
import sys

def read_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):  # Face
                faces.append(line.strip())
    return vertices, faces

def rotate_vertex_180_z(vertex):
    x, y, z = vertex
    # Rotate 180 degrees around the Z-axis
    x_rotated = -x
    y_rotated = -y
    return [x_rotated, y_rotated, z]

def write_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"{face}\n")

def main(input_file, output_file):
    vertices, faces = read_obj(input_file)
    rotated_vertices = [rotate_vertex_180_z(v) for v in vertices]
    write_obj(output_file, rotated_vertices, faces)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rotate_model.py <input_file.obj> <output_file.obj>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)