import os
import numpy as np
import argparse
import trimesh
import alg
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, choices=["bunny", "cow", "duck"], default="bunny")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    # load the mesh file and visualize
    mesh_path = "./meshes/%s.stl" % args.mesh
    mesh = trimesh.load(mesh_path)
    print("The mesh file was loaded by the path: %s" % os.path.abspath(mesh_path))
    print("Information of the mesh:")
    print("  Number of Faces: %d" % len(mesh.faces))
    print("  Number of Vertices: %d" % len(mesh.vertices))
    print("  The Center of Mass:", mesh.center_mass)
    print("\n")
    #_ = utils.plot_mesh(mesh)

    # Task 1: Primitive Wrenches Calculation 
    if args.task == 1:
        #grasp = np.random.choice(np.arange(len(mesh.faces)), size=3, replace=False)
        grasp = [0, 249, 484]
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the given grasp:")
        print(con_pts)
        utils.plot_grasp(mesh, grasp)
        W = alg.primitive_wrenches(mesh, grasp)
        utils.check_wrenches(mesh, grasp, W)

    # Task 2: Grasp Quality Evaluation 
    if args.task == 2:
        grasp = [0, 249, 484]
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the given grasp:")
        print(con_pts)
        utils.plot_grasp(mesh, grasp)
        Q = alg.eval_Q(mesh, grasp)
        print("The quality of the given grasp: %f \n" % Q)
    
    # Task 3: Sample a Stable Grasp
    elif args.task == 3:
        grasp, Q = alg.sample_stable_grasp(mesh, thresh=0.01)
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the stable grasp:")
        print(con_pts)
        print("The quality of the given grasp: %f \n" % Q)
        utils.plot_grasp(mesh, grasp)

    # Task 4: Optimize the Given Grasp
    elif args.task == 4:
        grasp = [80, 165, 444]
        traj = alg.optimize_grasp(mesh, grasp)
        print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        utils.plot_traj(mesh, traj)

    # Task 5: Sample and Optimize a Grasp under Reachability Constraint
    elif args.task == 5:
        traj = alg.optimize_reachable_grasp(mesh, r=1.0)
        print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        utils.plot_traj(mesh, traj)


# ### TESTING FOR REPORT

# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mesh", type=str, choices=["bunny", "cow", "duck"], default="bunny")
#     parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5])
#     args = parser.parse_args()

#     # load the mesh file and visualize
#     mesh_path = "./meshes/%s.stl" % args.mesh
#     mesh = trimesh.load(mesh_path)
#     print("The mesh file was loaded by the path: %s" % os.path.abspath(mesh_path))
#     print("Information of the mesh:")
#     print("  Number of Faces: %d" % len(mesh.faces))
#     print("  Number of Vertices: %d" % len(mesh.vertices))
#     print("  The Center of Mass:", mesh.center_mass)
#     print("\n")
#     #_ = utils.plot_mesh(mesh)

#     # Task 1: Primitive Wrenches Calculation 
#     if args.task == 1:
#         #grasp = np.random.choice(np.arange(len(mesh.faces)), size=3, replace=False)
#         grasp = [0, 249, 484]
#         print("The grasp:", grasp)
#         con_pts = utils.get_centroid_of_triangles(mesh, grasp)
#         print("The contact points of the given grasp:")
#         print(con_pts)
#         utils.plot_grasp(mesh, grasp)
#         W = alg.primitive_wrenches(mesh, grasp)
#         utils.check_wrenches(mesh, grasp, W)

#     # Task 2: Grasp Quality Evaluation 
#     if args.task == 2:
#         grasp = [0, 249, 484]
#         print("The grasp:", grasp)
#         con_pts = utils.get_centroid_of_triangles(mesh, grasp)
#         print("The contact points of the given grasp:")
#         print(con_pts)
#         utils.plot_grasp(mesh, grasp)
#         print(f"\n--- {args.mesh} ---")
#         for lmbd in [0.2, 0.5, 1.0]:
#             Q = alg.eval_Q(mesh, grasp, mu=0.2, n_edges=8, lmbd=lmbd)
#             print(f"  λ={lmbd}: Q={Q:.6f}")
    
#     # Task 3: Sample a Stable Grasp
#     elif args.task == 3:
#         # grasp, Q = alg.sample_stable_grasp(mesh, thresh=0.01)
#         # print("The grasp:", grasp)
#         # con_pts = utils.get_centroid_of_triangles(mesh, grasp)
#         # print("The contact points of the stable grasp:")
#         # print(con_pts)
#         # print("The quality of the given grasp: %f \n" % Q)
#         # utils.plot_grasp(mesh, grasp)

        
#         # Collect a larger batch for the histogram
#         print("\nSampling 100000 random grasps for histogram...")
#         n_faces = len(mesh.faces)
#         qualities = []
#         for _ in range(100000):
#             candidate = list(np.random.choice(n_faces, size=3, replace=False))
#             try:
#                 q = alg.eval_Q(mesh, candidate, lmbd=1.0)
#                 qualities.append(q)
#             except Exception:
#                 pass

#         plt.figure(figsize=(8, 5))
#         plt.hist(qualities, bins=40, edgecolor='black')
#         plt.axvline(x=0.01, color='r', linestyle='--', label='thresh=0.01')
#         plt.axvline(x=0.0,  color='orange', linestyle='--', label='Q=0 (stability boundary)')
#         plt.xlabel("L1 Grasp Quality Q")
#         plt.ylabel("Count")
#         plt.title("Distribution of Grasp Qualities (100000 random grasps, bunny)")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("task3_histogram.png", dpi=150)
#         plt.show()

#         stable_count = sum(1 for q in qualities if q > 0.01)
#         print(f"\nFraction above thresh=0.01: {stable_count}/{len(qualities)} = {stable_count/len(qualities):.2%}")

#     # Task 4: Optimize the Given Grasp
#     elif args.task == 4:
#         # grasp = [80, 165, 444]
#         # traj = alg.optimize_grasp(mesh, grasp)
#         # print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
#         # print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
#         # # utils.plot_traj(mesh, traj)
        
#         # # Count evaluations: at each step, local_optimal() evaluates the Cartesian
#         # # product of neighbor sets. Estimate this by inspecting neighbor set sizes.
#         # print(f"Trajectory length (steps accepted): {len(traj)}")
#         # total_evals = 0
#         # for g in traj[:-1]:  # for each step except the last (already at optimum)
#         #     nbr_sizes = [len(alg.find_neighbors(mesh, f, eta=1)) for f in g]
#         #     step_evals = 1
#         #     for s in nbr_sizes:
#         #         step_evals *= s
#         #     total_evals += step_evals
#         #     print(f"  grasp={g}, neighbor combo count ≈ {step_evals}")

#         # # One final call at the local optimum (to confirm no improvement)
#         # final_nbr_sizes = [len(alg.find_neighbors(mesh, f, eta=1)) for f in traj[-1]]
#         # final_evals = 1
#         # for s in final_nbr_sizes:
#         #     final_evals *= s
#         # total_evals += final_evals
#         # print(f"\nTotal evaluations ≈ {total_evals}")

#         # Collect a larger batch for the histogram
#         print("\nSampling 100000 random grasps for histogram...")
#         n_faces = len(mesh.faces)
#         qualities = []
#         for _ in range(100000):
#             candidate = list(np.random.choice(n_faces, size=3, replace=False))
#             try:
#                 q = alg.eval_Q(mesh, candidate, lmbd=1.0)
#                 qualities.append(q)
#             except Exception:
#                 pass
        
#         stable_count = sum(1 for q in qualities if q > 0.022)
#         print(f"\nFraction above thresh=0.022: {stable_count}/{len(qualities)} = {stable_count/len(qualities):.2%}")

#     # Task 5: Sample and Optimize a Grasp under Reachability Constraint
#     elif args.task == 5:
#         # traj = alg.optimize_reachable_grasp(mesh, r=0.5)
#         traj = alg.optimize_reachable_grasp(mesh, r=1.0)
#         print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
#         print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
#         utils.plot_traj(mesh, traj)
