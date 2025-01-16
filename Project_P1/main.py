from pagerank import compute_pagerank, plot_pagerank_comparison
from pagerank.networkx_utils import compare_pageranks, visualize_pagerank

def main():
    filenames = ['graph0.txt', 'graph1.txt', 'graph2.txt']
    
    for filename in filenames:
        try:
            # print(f"\nProcessing {filename}:")
            # ranking, eigenvalue, iterations = compute_pagerank(filename)
            # print(f"Dominant eigenvalue: {eigenvalue:.10f}")
            # print(f"Converged in {iterations} iterations")
            # print(f"Sum of ranks: {sum(rank for _, rank in ranking):.10f}")
            # print("\nPage rankings (descending order):")
            # print("Node\tRank")
            # print("-" * 20)

            # for node, rank in ranking:
            #     print(f"{node}\t{rank:.10f}")
            # plot_pagerank_comparison(filename)


            print(f"\nProcessing {filename}:")
            G, pagerank = compare_pageranks(filename)
            visualize_pagerank(G, pagerank, filename)
            


        except FileNotFoundError:
            print(f"Error: {filename} not found")
            continue

if __name__ == "__main__":
    main()