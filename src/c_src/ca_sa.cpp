#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stack>
#include <string>
#include <queue>
#include <unordered_set>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <unordered_map>
#include <iterator>
//#include <boost/geometry.hpp>
//#include <boost/geometry/geometries/polygon.hpp>
//#include <boost/geometry/geometries/box.hpp>

//namespace bg = boost::geometry;

//#typedef bg::model::d2::point_xy<double> point;
//typedef bg::model::polygon<point> polygon;
//typedef bg::model::box<point> box;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

enum edge_loc{
  NC = 0,
  LEFT=1,
  RIGHT=2,
  TOP=4,
  BOTTOM=8,
  TOP_LEFT = 5,
  TOP_RIGHT=6,
  BOTTOM_LEFT=9,
  BOTTOM_RIGHT=10,
};

using namespace std;

  
class Graph {
public:
    Graph(int numVertices) : V(numVertices), adjList(numVertices) {}

    void addEdge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u); // Undirected graph, so add edge in both directions
    }

    int countConnectedComponents() {
        std::vector<bool> visited(V, false);
        int count = 0;

        for (int v = 0; v < V; ++v) {
            if (!visited[v]) {
                DFS(v, visited);
                count++;
            }
        }

        return count;
    }

private:
    void DFS(int vertex, std::vector<bool>& visited) {
        visited[vertex] = true;

        for (int neighbor : adjList[vertex]) {
            if (!visited[neighbor]) {
                DFS(neighbor, visited);
            }
        }
    }

    int V; // Number of vertices
    std::vector<std::vector<int>> adjList; // Adjacency list
};

typedef struct hardblock {
    int id;
    int x;
    int y;
  int required_x;
  int required_y;
  int orig_width;
  int orig_height;
  edge_loc required_loc;
  edge_loc actual_loc;
  int cluster_id;
  int rectilinear_id;  
  bool preplaced;
  bool soft_preplaced;
  bool fixed_block;
  int width;
    int height;
    int rotate;
    int edge;
} HardBlock;

typedef struct terminal {
    int id;
    int x;
    int y;
} Terminal;

typedef struct node {
    int parent;
    int left_child;
    int right_child;
} Node;

typedef struct cost {
    int width;
    int height;
  double width_penalty;
  double height_penalty;
    double area;
    double wirelength;
    double R;
    double clusterP;
    double ws;
    double cost;
  int n_violated_edge_constraints;
  int extra_cluster_fragments;
  int extra_rectilinear_fragments;  
  double clustering_cost;
  double rectilinear_cost;  
  double preplaced_cost;
} Cost;

// Define a struct to store the grouping hash, key is the group id, and the values are list of blocks. 
typedef struct cluster {
    std::unordered_map<int, std::vector<int>> data;
} Cluster;


int num_hardblocks, num_terminals,num_clusters,num_clusters_rectilinear;
int num_nets, num_pins;
vector<HardBlock> hardblocks;
vector<HardBlock> initblocks;
vector<vector<int>> nets;
vector<Terminal> terminals;
Cluster cluster_hash;
std::vector<int> ar_list;

std::vector<double> wspace_list, wl_list, clust_list, cost_list;

double white_space_ratio;
double total_block_area;
double chip_ar;//desired chip's aspect ratio
double alpha;//wl factor
double floorplan_boundaries_weight;
double _beta;//edge penalty factor
double beta_cluster;//cluster penalty factor
double beta_rectilinear;//rectilinear penalty factor
double beta_preplaced;//preplaced penalty factor
bool inverse;//to enable negative cost 
double ws_thresh;//to ensure smaller steps of ws change
double ar_increment;

double delta1, delta2;
int T_step;
double T0 = 0.001;
double constraint_fixing_prob = 0.0;
double tlimit, step_limit;//time-out (either using time or step counter)
//alpha + beta <= 1
// target area = total block area * (1 + white space ratio)
double area_target;
// area, wire length normalization = initial area, wirelength
double area_norm = 0, wl_norm = 0, edge_norm = 0, cluster_norm = 0;
// fixed outline max x coordinate
int W, H;
bool ar_search;
// b*-tree
int root_block = -1;
vector<Node> btree;

//to return to python API
std::vector<std::vector<int>> current_btree;
std::vector<std::vector<int>> current_loc;


//State trajectory
std::vector<std::vector<std::vector<int>>> state_trajectory;

//Loc trajectory
std::vector<std::vector<std::vector<int>>> location_trajectory;


//Action trajectory
std::vector<std::vector<int>> action_trajectory;

//Reward trajectory
std::vector<std::vector<double>> reward_trajectory;


//hardblock positions and dimensions
std::vector<std::vector<int>> blockPos;

// horizontal contour
vector<int> contour;

//placed list tracker
vector<int> placed_list;

bool in_fixed_outline;
// floorplan with minimum cost
int min_cost_root_block;
Cost min_cost;
vector<HardBlock> min_cost_floorplan;
vector<Node> min_cost_btree;
// floorplan in fixed outline with minimum cost
int min_cost_root_block_fixed_outline;
Cost min_cost_fixed_outline;
vector<HardBlock> min_cost_floorplan_fixed_outline;
vector<Node> min_cost_btree_fixed_outline;


//edge groups derived from the physical locations after graph-traversal
vector<int> northG, southG, eastG, westG;
int graph_sw, graph_se, graph_nw, graph_ne;
//Contours for storing the current edge shapes
vector<int> Rcontour;
vector<int> Lcontour;
vector<int> Tcontour;
vector<int> Bcontour;


//from constraints - inputs
vector<int> blocks_east_boundary, blocks_west_boundary, blocks_north_boundary, blocks_south_boundary;
vector<int> blocks_core_region, blocks_edge_region, blocks_corner_region;
int sw_corner, se_corner, nw_corner, ne_corner;
bool enable_floorplan_boundaries;
int floorplan_width;
int floorplan_height;
  
template <class T>
inline T abs(T val)
{
  return val > 0 ? val : -val;
}

void ReadHardBlocks(std::vector<std::vector<int>> hardblocks_list,bool initialize)
{

    num_hardblocks = hardblocks_list.size();
    num_clusters = 0;
    num_clusters_rectilinear = 0;
    total_block_area = 0;
    hardblocks = vector<HardBlock>(num_hardblocks);
    initblocks = vector<HardBlock>(num_hardblocks);
    for (int i = 0; i < num_hardblocks; i++) {

        int width = hardblocks_list[i][0];
        int height = hardblocks_list[i][1];
        edge_loc required_loc = (edge_loc) hardblocks_list[i][2];
        int cluster_id = hardblocks_list[i][3];
        int rectilinear_id = hardblocks_list[i][4];        
        bool fixed_block = (bool)hardblocks_list[i][5];
        bool preplaced = (bool)hardblocks_list[i][6];
        int x_preplaced = hardblocks_list[i][7];
        int y_preplaced = hardblocks_list[i][8];                
        num_clusters = max(num_clusters,cluster_id);
        num_clusters_rectilinear = max(num_clusters_rectilinear,rectilinear_id);        
        
        hardblocks[i].id = i;
        hardblocks[i].x = hardblocks[i].required_x = (preplaced ? x_preplaced : -1);
        hardblocks[i].y = hardblocks[i].required_y = (preplaced ? y_preplaced : -1);
        hardblocks[i].width = width;
        hardblocks[i].height = height;
        hardblocks[i].orig_width = width;
        hardblocks[i].orig_height = height;
        hardblocks[i].required_loc = required_loc;
        hardblocks[i].cluster_id = cluster_id;
        hardblocks[i].rectilinear_id = rectilinear_id;        
        hardblocks[i].fixed_block = fixed_block;
        hardblocks[i].preplaced = preplaced;                                
        hardblocks[i].soft_preplaced = false;
        
        hardblocks[i].rotate = 0;
        if(initialize)
          {
            //will not be changed
            initblocks[i].id = i;
            initblocks[i].x = initblocks[i].required_x = (preplaced ? x_preplaced : -1);
            initblocks[i].y = initblocks[i].required_y = (preplaced ? y_preplaced : -1);
            initblocks[i].width = width;
            initblocks[i].height = height;
            initblocks[i].rotate = 0;
            initblocks[i].required_loc = required_loc;
            initblocks[i].cluster_id = cluster_id;
            initblocks[i].rectilinear_id = rectilinear_id;
            initblocks[i].fixed_block = fixed_block;
            initblocks[i].preplaced = preplaced;
            initblocks[i].soft_preplaced = false;            
          }

        
        total_block_area += width * height;
    }

    area_target = total_block_area * (1 + white_space_ratio);
    //FROM OUTLINE
    //W = sqrt(area_target);//for square chip
    //H = sqrt(area_target);// for square chip

    //non-square SOC chip_ar = Chip Width / Chip Height
    H = int(sqrt(area_target/chip_ar));
    W = int(area_target / H);

    cout << "Area:             " << total_block_area << '\n';
    cout << "Target Area:      " << area_target << '\n';
    cout << "W:                " << W << '\n';
    cout << "H:                " << H << '\n';
    cout << "Chip AR:          " << chip_ar << '\n';  
    cout << '\n';

}






void ReadNets(std::vector<std::vector<int>> nets_list)
{
    //nets = vector<vector<int>>(num_nets);
    nets = nets_list;

}



void ReadTerminals(std::vector<std::vector<int>> terminals_list)
{

    num_terminals = terminals_list.size();

    terminals = vector<Terminal>(num_terminals);
    for (int i = 0; i < num_terminals; i++) {
        terminals[i].id = i;
        terminals[i].x = terminals_list[i][0];
        terminals[i].y = terminals_list[i][1];
    }


}








void BuildInitBtree()
{
    btree = vector<Node>(num_hardblocks);
    queue<int> bfs;
    vector<int> inserted(num_hardblocks, 0);

    root_block = rand() % num_hardblocks;
    btree[root_block].parent = -1;
    bfs.push(root_block);
    inserted[root_block] = 1;

    int left = num_hardblocks - 1;
    while (!bfs.empty()) {
        int parent = bfs.front();
        bfs.pop();
        int left_child = -1, right_child = -1;
        if (left > 0) {
            do {
                left_child = rand() % num_hardblocks;
            } while (inserted[left_child]);
            btree[parent].left_child = left_child;
            bfs.push(left_child);
            inserted[left_child] = 1;
            left--;
            if (left > 0) {
                do {
                    right_child = rand() % num_hardblocks;
                } while (inserted[right_child]);
                btree[parent].right_child = right_child;
                bfs.push(right_child);
                inserted[right_child] = 1;
                left--;
            }
        }
        btree[parent].left_child = left_child;
        btree[parent].right_child = right_child;
        if (left_child != -1)
            btree[left_child].parent = parent;
        if (right_child != -1)
            btree[right_child].parent = parent;
    }
}



void printDFSub(int current_node_id)
{

    if (btree[current_node_id].left_child != -1) {
        cout << "LEFT Edge: " << current_node_id << ":" << btree[current_node_id].left_child << "\n";
        printDFSub(btree[current_node_id].left_child);
    }

    if (btree[current_node_id].right_child != -1) {
        cout << "RIGHT Edge: " << current_node_id << ":" << btree[current_node_id].right_child << "\n";
        printDFSub(btree[current_node_id].right_child);
    }

}


void printDF() 
{
    cout << "Printing DFS Tree " << "\n";
    int current_node_id = root_block;

    if (btree[current_node_id].left_child != -1) {
        cout << "LEFT Edge: " << current_node_id << ":" << btree[current_node_id].left_child << "\n";
        printDFSub(btree[current_node_id].left_child);
    }

    if (btree[current_node_id].right_child != -1) {
        cout << "RIGHT Edge: " << current_node_id << ":" << btree[current_node_id].right_child << "\n";
        printDFSub(btree[current_node_id].right_child);
    }

}


void LoadBTree(std::vector<std::vector<int>>  tree_list){
    cout << "Loading binary tree " << "\n";
    btree = vector<Node>(num_hardblocks);
    vector<int> inserted(num_hardblocks, 0);

    int n1, n2, edge_type;
    //cout << "TREE Initializating \n";
    //initialize childs for each node
    for (int i = 0; i < num_hardblocks; i++) {
        btree[i].left_child = -1;
        btree[i].right_child = -1;
        btree[i].parent = -1;
    }

    int num_edges = tree_list.size();
    for (int i = 0; i < num_edges; i++) {
        edge_type = tree_list[i][2] ;
        n1 = tree_list[i][0];
        n2 = tree_list[i][1];
        //cout << "Edge : " << i << " :: " << edge_type << " " << n1 << " " << n2 << "\n"; 

        if (edge_type == 0) {
            //cout << "LEFT EDGE: " << n1 << " : " << n2 << "\n";
            inserted[n1] = 1;
            inserted[n2] = 1;
            btree[n1].left_child = n2;
            btree[n2].parent = n1;
            if (i == 0) {
                root_block = n1;
            }
        } else if (edge_type == 1) {
            //cout << "RIGHT EDGE: " << n1 << " : " << n2 << "\n";
            inserted[n1] = 1;
            inserted[n2] = 1;
            btree[n1].right_child = n2;
            btree[n2].parent = n1;
            if (i == 0) {
                root_block = n1;
            }
        } else {
            cout << "[ERROR]: Invalid edge found in the tree file: \n";
        }
    }



    bool hasZero = false;

    for (int number : inserted) {
        //cout << "Inserted element " << number << "\n";
        if (number == 0) {
            hasZero = true;
            break;  // Exit the loop early if "0" is found
        }
    }

    if (hasZero) {
        std::cout << "TREE is INVALID..." << std::endl;
    } else {
        std::cout << "[SUCCESS] TREE IS VALID..." << std::endl;
    }



}





void BuildPreExistBTree(std::vector<std::vector<int>> tree_list){
    btree = vector<Node>(num_hardblocks);
    vector<int> inserted(num_hardblocks, 0);
    if(tree_list.size() != num_hardblocks - 1)
      {
        py::print("Error : Invalid tree with only ",tree_list.size()," entries. Expecting ",num_hardblocks - 1);
        return ;
      }

    int temp1, temp2, temp3;

    //initialize childs for each node
    for (int i = 0; i < num_hardblocks; i++) {
        btree[i].left_child = -1;
        btree[i].right_child = -1;
        btree[i].parent = -1;
    }

    for (int i = 0; i < num_hardblocks-1; i++) {
      temp1 = tree_list[i][0];
      temp2 = tree_list[i][1];
      temp3 = tree_list[i][2];
      
      if (temp3 == 0) {
        //cout << "LEFT EDGE: " << temp1 << " : " << temp2 << "\n";
        inserted[temp1] = 1;
        inserted[temp2] = 1;
        btree[temp1].left_child = temp2;
        btree[temp2].parent = temp1;
        if (i == 0) {
          root_block = temp1;
        }
      } else if (temp3 == 1) {
        //cout << "RIGHT EDGE: " << temp1 << " : " << temp2 << "\n";
        inserted[temp1] = 1;
        inserted[temp2] = 1;
        btree[temp1].right_child = temp2;
        btree[temp2].parent = temp1;
        if (i == 0) {
          root_block = temp1;
        }
      } else {
        cout << "[ERROR]: Invalid edge found in the tree file: \n";
      }
    }



    bool hasZero = false;

    for (int number : inserted) {
        //cout << "Inserted element " << number << "\n";
        if (number == 0) {
            hasZero = true;
            break;  // Exit the loop early if "0" is found
        }
    }

    if (hasZero) {
      py::print("Tree is invalid");
        std::cout << "TREE is INVALID..." << std::endl;
    } else {
        std::cout << "[SUCCESS] TREE IS VALID..." << std::endl;
    }



}



//Updated to not use contours
void BtreePreorderTraverseU(int cur_node, bool flag,bool place_root = false)
{
  if(!hardblocks[cur_node].preplaced)
    {    
      if(place_root) //we are placing the root
        hardblocks[cur_node].x = 0;
      else
        {
          int parent = btree[cur_node].parent;
          if (flag == false) {//LEFT child, right neighbor
            hardblocks[cur_node].x = hardblocks[parent].x + hardblocks[parent].width;
          } else {
            hardblocks[cur_node].x = hardblocks[parent].x;
          }
        }

      int x_start = hardblocks[cur_node].x;
      int x_end = x_start + hardblocks[cur_node].width;
      
      std::map<int,int> blocking_spans; //ordered by lower edge
      for (int element : placed_list) {
        int xleft = hardblocks[element].x;
        int xright = xleft + hardblocks[element].width;
        if (x_start < xright && x_end >xleft)
          {
            
            blocking_spans[hardblocks[element].y] = max(hardblocks[element].y + hardblocks[element].height,
                                                        blocking_spans[hardblocks[element].y]);
          }
      }
      
      int y_placed = 0;
      int cur_height = hardblocks[cur_node].height;
      
      for (auto vert_span : blocking_spans)
        {
          //vert_span.first is lower edge, vert_span.second is top edge
          //          py::print("blocking span for block ",cur_node," with y_placed ",y_placed, " and height " ,cur_height, " is ",vert_span.first, " to ",vert_span.second);

          if(vert_span.first >= y_placed + cur_height)
            break;
          if(vert_span.second > y_placed)
            y_placed = vert_span.second;
        }
      
      hardblocks[cur_node].y = y_placed;
  
      placed_list.push_back(cur_node);
    }
  
  if (btree[cur_node].left_child != -1)
    {
      current_btree.push_back({{cur_node, btree[cur_node].left_child, 0}});
      BtreePreorderTraverseU(btree[cur_node].left_child, false);//LEFT == 0 == FALSE
    }
  if (btree[cur_node].right_child != -1)
    {
      current_btree.push_back({{cur_node, btree[cur_node].right_child, 1}});
      BtreePreorderTraverseU(btree[cur_node].right_child, true);//RIGHT == 1 == TRUE
    }
}




std::vector<std::vector<int>> getBlockPos(){ 
  blockPos.clear();
  for (int i = 0; i < num_hardblocks; i++) {
    std::vector<int> current_pos;
    current_pos.push_back(hardblocks[i].x);
    current_pos.push_back(hardblocks[i].y);
    current_pos.push_back(hardblocks[i].width );
    current_pos.push_back(hardblocks[i].height);
    current_pos.push_back(hardblocks[i].actual_loc );
    blockPos.push_back(current_pos);
  }
  return blockPos;
}


std::vector<std::vector<int>> getProblem(){ 
  std::vector<std::vector<int>> problem;
  for (int i = 0; i < num_hardblocks; i++) {
    std::vector<int> current_pos;
    current_pos.push_back(hardblocks[i].width);
    current_pos.push_back(hardblocks[i].height);
    current_pos.push_back(hardblocks[i].required_loc );
    current_pos.push_back(hardblocks[i].cluster_id);
    current_pos.push_back(hardblocks[i].rectilinear_id );
    current_pos.push_back(int(hardblocks[i].fixed_block));
    current_pos.push_back(int(hardblocks[i].preplaced));

    current_pos.push_back(hardblocks[i].preplaced ? hardblocks[i].x : -1);
    current_pos.push_back(hardblocks[i].preplaced ? hardblocks[i].y : -1);    
    problem.push_back(current_pos);
  }
  return problem;
}

void AddEdgeLocs()
{
  int left_x,bottom_y,right_x,top_y;
  for (int i = 0; i < num_hardblocks; i++)
    {
      if(i==0)
        {
          left_x = hardblocks[i].x + hardblocks[i].width;
          right_x = hardblocks[i].x;
          bottom_y = hardblocks[i].y + hardblocks[i].height;
          top_y = hardblocks[i].y;
        }
      else
        {
          left_x = min(left_x,hardblocks[i].x + hardblocks[i].width);
          right_x = max(right_x,hardblocks[i].x);
          bottom_y = min(bottom_y,hardblocks[i].y + hardblocks[i].height);
          top_y = max(top_y,hardblocks[i].y);
        }
    }
    for (int i = 0; i < num_hardblocks; i++)
      {
        edge_loc horiz = NC;
        edge_loc vert = NC;

        if(hardblocks[i].x == 0)
          horiz = LEFT;

        if(hardblocks[i].x +  hardblocks[i].width  > right_x)
          horiz = RIGHT;

        if(hardblocks[i].y   == 0)
          vert = BOTTOM;


        if(hardblocks[i].y + hardblocks[i].height  > top_y)
          vert = TOP;

        hardblocks[i].actual_loc = (edge_loc)(horiz + vert);
        
      }
    
    
}



void BtreeToFloorplan()
{
    //std::vector<std::vector<int>> current_btree;
    //cout << "Generating layout from btree \n";
    current_btree.clear();

    current_loc.clear();

    //clear the placed_list vector
    placed_list.clear();
    for (int i = 0; i < num_hardblocks; i++)
      if(hardblocks[i].preplaced)
        placed_list.push_back(i);
            
    
    BtreePreorderTraverseU(root_block, false,true);

    //get block positions
    current_loc = getBlockPos();

    AddEdgeLocs();
    
}

bool BlocksTouching(int i,int j)
{
  bool x_overlap_left = (hardblocks[i].x  >= hardblocks[j].x) &&
    (hardblocks[i].x <= hardblocks[j].x + hardblocks[j].width);
  bool x_overlap_right = (hardblocks[i].x+hardblocks[i].width  >= hardblocks[j].x) &&
    (hardblocks[i].x + hardblocks[i].width <= hardblocks[j].x + hardblocks[j].width);

  bool x_overlap = x_overlap_left || x_overlap_right;


  bool y_overlap_bot = (hardblocks[i].y  >= hardblocks[j].y) &&
    (hardblocks[i].y <= hardblocks[j].y + hardblocks[j].height);
  bool y_overlap_top = (hardblocks[i].y+hardblocks[i].height  >= hardblocks[j].y) &&
    (hardblocks[i].y + hardblocks[i].height <= hardblocks[j].y + hardblocks[j].height);
  
  bool y_overlap = y_overlap_bot || y_overlap_top;
  
  return x_overlap && y_overlap;
}

bool BlockTouchingCluster(int i,const vector<int> & cluster)
{
  //  py::print("Checking if block is touching cluster ",i,"\n");  
  for(int b=0;b < cluster.size();++b)
    if(BlocksTouching(i,cluster[b]))
      return true;
  return false;
}

vector<vector<int>> MergeClusters(const vector<vector<int>> &clusters,
                                  const vector<int> & merge_locs)
{
  vector<vector<int> > merge_result;
  if(merge_locs.size() < 2)
    py::print("ERROR : cluster merge locations less than 2 ");
  
  if(merge_locs.back() >= clusters.size())
    py::print("ERROR : merge locs entry exceeds number of clusters  ",merge_locs.back()," ",clusters.size());

  vector<int> merged_cluster;
  for(int m=0;m < merge_locs.size();++m)
    merged_cluster.insert(merged_cluster.end(),clusters[merge_locs[m]].begin(),clusters[merge_locs[m]].end());

  merge_result.push_back(merged_cluster);

  int merge_idx = 0;
  for(int c=0;c < clusters.size(); ++ c)
    if(merge_idx >= merge_locs.size()  || c != merge_locs[merge_idx])
      merge_result.push_back(clusters[c]);
    else
      {
        merge_idx ++;
      }
  
  return merge_result;

}



vector<vector<vector<int> > > getClusterData(bool rectilinear_clusters)
{
  //py::print("getting cluster data for n_clusters ",num_clusters,"\n");
  vector<vector<vector<int> > > cluster_data(rectilinear_clusters ? num_clusters_rectilinear : num_clusters);
  
  for (int i = 0; i < num_hardblocks; i++) {
    int clustering_id = rectilinear_clusters ? hardblocks[i].rectilinear_id : hardblocks[i].cluster_id;
    if (clustering_id != 0)
      {
        int cluster_idx = clustering_id - 1;
        vector <int> touching_clusters;
        for(int c=0;c<cluster_data[cluster_idx].size();++c)
          if(BlockTouchingCluster(i,cluster_data[cluster_idx][c]))
            touching_clusters.push_back(c);
        
        if(touching_clusters.size() == 0) //create new de-factor cluster
          cluster_data[cluster_idx].push_back(vector<int>(1,i));
        else if (touching_clusters.size() == 1) // add to the cluster
          cluster_data[cluster_idx][touching_clusters[0]].push_back(i);
        else //i is a bridge between 2 previously separate clusters
          {
            auto merged_cluster = MergeClusters(cluster_data[cluster_idx],
                                                touching_clusters);
            merged_cluster[0].push_back(i); //0 is always the index of merged cluster
            cluster_data[cluster_idx] = merged_cluster;
          }
      }

  }
  return cluster_data;
}
  

void getClusterCost(bool rectilinear_clusters,double &clustering_cost,int & extra_fragments)
{
    vector<vector<vector<int>>> cluster_data = getClusterData(rectilinear_clusters);
    int extra_cluster_fragments = 0;
    int n_cluster_blocks = 0;
    for(int cluster_id = 0;cluster_id < cluster_data.size();++cluster_id)
      {
        extra_cluster_fragments += (cluster_data[cluster_id].size() - 1);
        for(int c= 0;c < cluster_data[cluster_id].size();++c)
          n_cluster_blocks += cluster_data[cluster_id][c].size();
      }
    if(n_cluster_blocks == 0)
      clustering_cost = 0;
    else
      clustering_cost = (extra_cluster_fragments * 1.0 / n_cluster_blocks);      
    extra_fragments = extra_cluster_fragments;

}


Cost CalculateCost()
{
    BtreeToFloorplan();//convert tree to layout

    int width = 0, height = 0;
    double block_area = 0.0;
    for (int i = 0; i < num_hardblocks; i++) {
        block_area += (int(hardblocks[i].width)*int(hardblocks[i].height));
        if (hardblocks[i].x + hardblocks[i].width > width)
            width = hardblocks[i].x + hardblocks[i].width;
        if (hardblocks[i].y + hardblocks[i].height > height)
            height = hardblocks[i].y + hardblocks[i].height;
    }

    // area of current floorplan
    double floorplan_area = width * height;
    // aspect ratio of current floorplan
    double R = (double)height / width;

    // half perimeter wire length
    double wirelength = 0;
    for (const vector<int> &net : nets) {
        int x_min = width + 1, x_max = 0;
        int y_min = height + 1, y_max = 0;
        for (const int pin : net) {
            if (pin < num_hardblocks) {
                int x_center = hardblocks[pin].x + hardblocks[pin].width / 2;
                int y_center = hardblocks[pin].y + hardblocks[pin].height / 2;
                if (x_center < x_min)
                    x_min = x_center;
                if (y_center < y_min)
                    y_min = y_center;
                if (x_center > x_max)
                    x_max = x_center;
                if (y_center > y_max)
                    y_max = y_center;
            }
            else {
                const Terminal &t = terminals[pin - num_hardblocks];
                if (t.x < x_min)
                    x_min = t.x;
                if (t.y < y_min)
                    y_min = t.y;
                if (t.x > x_max)
                    x_max = t.x;
                if (t.y > y_max)
                    y_max = t.y;
            }
        }

        wirelength += (x_max - x_min) + (y_max - y_min);
    }

    double preplaced_cost = 0;
    for (int i = 0; i < num_hardblocks; i++)
      if(hardblocks[i].preplaced || hardblocks[i].soft_preplaced)
        preplaced_cost += abs(hardblocks[i].x - hardblocks[i].required_x) + abs(hardblocks[i].y - hardblocks[i].required_y) ;
    

    Cost c;
    c.width = width;
    c.height = height;
    c.area = floorplan_area;
    c.wirelength = wirelength;
    c.R = R;
    c.ws = (floorplan_area - block_area)*100.0/(floorplan_area);
    c.preplaced_cost = preplaced_cost;
    
    // set normalization to initial floorplan area and wirelength (also added edge norm)
    if (area_norm == 0) {
        area_norm = floorplan_area;
    }
    if (wl_norm == 0) {
        wl_norm = wirelength;
        //cout << "INITIAL WIRELENGTH COST = " << wl_norm << "\n";
    }

    //    py::print("INITIAL AREA COST = ",area_norm," ",c.area,"\n");

    
  ///Calculate edge violations
    int n_violated_edge_constraints  = 0;
    int n_edge_constraints = 0;
    for (int i = 0; i < num_hardblocks; i++)
      {
        if(hardblocks[i].required_loc != NC)
          {
            n_edge_constraints += 1;
            if(hardblocks[i].actual_loc != hardblocks[i].required_loc)
              n_violated_edge_constraints += 1;
          }
      }
    c.n_violated_edge_constraints = n_violated_edge_constraints;
    double edge_violation_cost = 0.0;
    if(n_edge_constraints > 0)
      edge_violation_cost = (n_violated_edge_constraints * 1.0) / n_edge_constraints;
    ////

    

    


    getClusterCost(false,c.clustering_cost,c.extra_cluster_fragments);
    getClusterCost(true,c.rectilinear_cost,c.extra_rectilinear_fragments);    
    
            
          
      
    
    double area_cost = c.ws / 100.0;
    //double area_cost = c.area / area_norm;    
    double wl_cost = c.wirelength / wl_norm;
    double R_cost = 0;//(1 - R) * (1 - R);//made 0, to remove square AR for chip-level
    double width_penalty = 0;
    double height_penalty = 0;


    if (enable_floorplan_boundaries && width > floorplan_width) {
        width_penalty = ((double)width / floorplan_width);
    }
    if (enable_floorplan_boundaries &&height > floorplan_height) {
        height_penalty = ((double)height / floorplan_height);
    }
    c.width_penalty = width_penalty * floorplan_boundaries_weight;
    c.height_penalty = height_penalty * floorplan_boundaries_weight;
    
    c.cost = (area_cost)*(1 - alpha) + (width_penalty + height_penalty) * floorplan_boundaries_weight + wl_cost*(alpha) + _beta * edge_violation_cost
      + beta_cluster * c.clustering_cost + beta_rectilinear * c.rectilinear_cost +  (c.preplaced_cost * beta_preplaced / (width + height));
    if (inverse) {
        c.cost = 1 / c.cost ;
    }

#ifdef DEBUG
    cout << "Width:      " << c.width << '\n';
    cout << "Height:     " << c.height << '\n';
    cout << "Area:       " << c.area << '\n';
    cout << "Wirelength: " << c.wirelength << '\n';
    cout << "R:          " << c.R << '\n';
    cout << "Cost:       " << c.cost << '\n';
    cout << '\n';
#endif

    return c;
}

void hard_preplace_constraints(bool hard)
{
  for (int i = 0; i < num_hardblocks; i++)
    {
      if(hard && hardblocks[i].soft_preplaced)
        {
          hardblocks[i].preplaced = true;
          hardblocks[i].soft_preplaced = false;
        }
      if(!hard && hardblocks[i].preplaced)
        {
          hardblocks[i].soft_preplaced = true;
          hardblocks[i].preplaced = false;
        }
    }
}


void ARAdjust(int node,double suggested_adjust = 0.0)
{
    int temp = hardblocks[node].width;
    int init_area = hardblocks[node].orig_width * hardblocks[node].orig_height;
    double ar = ((double) hardblocks[node].width) / hardblocks[node].height;

    double ar_adjust;
    if(suggested_adjust == 0.0)
      {
        if(ar >= 3.0)
          ar_adjust = -ar_increment;
        else if (ar <= (1/3.0))
          ar_adjust = ar_increment;
        else
          if((rand() % 2) == 0)
            ar_adjust = ar_increment;
          else
            ar_adjust = -ar_increment;
      }
    else
      ar_adjust = suggested_adjust;
        
    double new_ar = ar + ar_adjust;
    int new_h  = int(std::sqrt(init_area / new_ar) + 0.5);
    if (new_h == 0) {
      new_h = 1;
    }
    int new_w = int((double)init_area / new_h + 0.5);
    if (new_w == 0) {
      new_w = 1;
    }

    hardblocks[node].width = new_w;
    hardblocks[node].height = new_h;

    
}

void Rotate(int node)
{
    int temp = hardblocks[node].width;
    hardblocks[node].width = hardblocks[node].height;
    hardblocks[node].height = temp;
    hardblocks[node].rotate = 1 - hardblocks[node].rotate;
}



void Swap(int node1, int node2)
{
    // swap parent
    //cout << "Committed swap operation " << "\n";
    int node1_parent = btree[node1].parent;
    int node2_parent = btree[node2].parent;

    if (node1_parent == node2_parent) {
        int left_child = btree[node1_parent].left_child;
        int right_child = btree[node1_parent].right_child;
        btree[node1_parent].left_child = right_child;
        btree[node1_parent].right_child =  left_child;
        //cout << "[DEBUG 0 for same parent] " << node1_parent << " - " << btree[node1_parent].left_child << " : " << btree[node1_parent].right_child << "\n";

    } else {
        //update node1's parent's left and right child to node2
        if (node1_parent != -1) {
            if (btree[node1_parent].left_child == node1)
                btree[node1_parent].left_child = node2;
            else if (btree[node1_parent].right_child == node1)
                btree[node1_parent].right_child = node2;
            else {
                cout << "[Error] node not parent's child\n";
                exit(1);
            }
        }

        //cout << "[DEBUG 0 for same parent] " << node1_parent << " - " << btree[node1_parent].left_child << " : " << btree[node1_parent].right_child << "\n";
        //update node2's parent's left and right child to node1
        if (node2_parent != -1) {
            if (btree[node2_parent].left_child == node2)
                btree[node2_parent].left_child = node1;
            else if (btree[node2_parent].right_child == node2)
                btree[node2_parent].right_child = node1;
            else {
                cout << "[Error] node not parent's child\n";
                exit(1);
            }
        }
    }
    //cout << "[DEBUG 1 for same parent] " << node1_parent << " - " << btree[node1_parent].left_child << " : " << btree[node1_parent].right_child << "\n";
    //update parent's if they are from diff parents
    if (node1_parent != node2_parent) {
        btree[node1].parent = node2_parent;
        btree[node2].parent = node1_parent;
    } else {
        //cout << "[DEBUG 2 for same parent] " << node1_parent << " - " << node2_parent << " *** " << btree[node1_parent].left_child << " : " << btree[node1_parent].right_child << "\n";
    }
    //else if (btree[node1].parent != btree[node2].parent) {//swap left and right
    //}
    // swap children
    int node1_left_child = btree[node1].left_child;
    int node1_right_child = btree[node1].right_child;
    int node2_left_child = btree[node2].left_child;
    int node2_right_child = btree[node2].right_child;
    btree[node1].left_child = node2_left_child;
    btree[node1].right_child = node2_right_child;
    btree[node2].left_child = node1_left_child;
    btree[node2].right_child = node1_right_child;

    //update child to parent relation 
    if (btree[node1].left_child != -1)
        btree[btree[node1].left_child].parent = node1;
    if (btree[node1].right_child != -1)
        btree[btree[node1].right_child].parent = node1;
    if (btree[node2].left_child != -1)
        btree[btree[node2].left_child].parent = node2;
    if (btree[node2].right_child != -1)
        btree[btree[node2].right_child].parent = node2;

    // node1, node2 are parent and child
    if (btree[node1].parent == node1)
        btree[node1].parent = node2;
    else if (btree[node1].left_child == node1)
        btree[node1].left_child = node2;
    else if (btree[node1].right_child == node1)
        btree[node1].right_child = node2;

    if (btree[node2].parent == node2)
        btree[node2].parent = node1;
    else if (btree[node2].left_child == node2)
        btree[node2].left_child = node1;
    else if (btree[node2].right_child == node2)
        btree[node2].right_child = node1;

    // root block may change
    if (node1 == root_block)
        root_block = node2;
    else if (node2 == root_block)
        root_block = node1;
}

//regular flow
void Move(int node, int to_node,bool favor_left = false,bool favor_right = false)
{
    // delete
    if (btree[node].left_child == -1 && btree[node].right_child == -1) {
        // no children
        int parent = btree[node].parent;
        if (btree[parent].left_child == node)
            btree[parent].left_child = -1;
        else if (btree[parent].right_child == node)
            btree[parent].right_child = -1;
        else {
            cout << "[Error] node not parent's child\n";
            exit(1);
        }
    }
    else if (btree[node].left_child != -1 && btree[node].right_child != -1) {
        // two children
        do {
            bool move_left;
            if (btree[node].left_child != -1 && btree[node].right_child != -1)
                move_left = rand() % 2 == 0;
            else if (btree[node].left_child != -1)
                move_left = true;
            else
                move_left = false;
            
            if (move_left)
                Swap(node, btree[node].left_child);
            else
                Swap(node, btree[node].right_child);
        } while (btree[node].left_child != -1 || btree[node].right_child != -1);

        int parent = btree[node].parent;
        if (btree[parent].left_child == node)
            btree[parent].left_child = -1;
        else if (btree[parent].right_child == node)
            btree[parent].right_child = -1;
        else {
            cout << "[Error] node not parent's child\n";
            exit(1);
        }
    }
    else {
        // one child
        int child;
        if (btree[node].left_child != -1)
            child = btree[node].left_child;
        else
            child = btree[node].right_child;

        int parent = btree[node].parent;
        if (parent != -1) {
            if (btree[parent].left_child == node)
                btree[parent].left_child = child;
            else if (btree[parent].right_child == node)
                btree[parent].right_child = child;
            else {
                cout << "[Error] [one child] node not parent's child\n";
                exit(1);
            }
        }

        btree[child].parent = parent;

        // root block may change
        if (node == root_block)
            root_block = child;
    }

    // insert (**corrected the flow for various rand values**)
    int random_left_right;
    if(favor_left)
      random_left_right = 0;
    else if(favor_right)
      random_left_right = 3;
    else
      random_left_right = rand() % 4;
      
    int child;
    if (random_left_right == 0) {
        child = btree[to_node].left_child;
        btree[node].left_child = child;
        btree[node].right_child = -1;
        btree[to_node].left_child = node;
    }
    else if (random_left_right == 1) {
        child = btree[to_node].right_child;
        btree[node].left_child = child;
        btree[node].right_child = -1;
        btree[to_node].right_child = node;
    }
    else if (random_left_right == 2) {
        child = btree[to_node].left_child;
        btree[node].left_child = -1;
        btree[node].right_child = child;
        btree[to_node].left_child = node;
    }
    else {
        child = btree[to_node].right_child;
        btree[node].left_child = -1;
        btree[node].right_child = child;
        btree[to_node].right_child = node;
    }
    btree[node].parent = to_node;
    if (child != -1)
        btree[child].parent = node;
}

void fix_edge_violation(int node)
{
  if(hardblocks[node].required_loc == NC)
    {
      py::print("Node does not have edge constraint ",node,"\n");
      return;
    }
  if(hardblocks[node].required_loc == hardblocks[node].actual_loc)
    {
      py::print("Node already satisfies edge constraint ",node,hardblocks[node].actual_loc,"\n");
      return;
    }

  vector<int> swappable_nodes;
  vector<int> non_swappable_nodes;
  for (int i = 0; i < num_hardblocks; i++)
    {
      if(hardblocks[i].actual_loc == hardblocks[node].required_loc)
        {
          if(hardblocks[i].actual_loc == hardblocks[i].required_loc)
            non_swappable_nodes.push_back(i);
          else
            swappable_nodes.push_back(i);
        }
    }
  if(swappable_nodes.size() > 0)
    {
      int node_to_swap = rand() %swappable_nodes.size();
      Swap(node,swappable_nodes[node_to_swap]);
    }
  else if(non_swappable_nodes.size() > 0)
    {
      int node_to_move_to = rand() % non_swappable_nodes.size();
      if(hardblocks[node].required_loc == LEFT || hardblocks[node].required_loc == RIGHT)
        Move(node,non_swappable_nodes[node_to_move_to],false,true);
      else if(hardblocks[node].required_loc == BOTTOM || hardblocks[node].required_loc == TOP)
        Move(node,non_swappable_nodes[node_to_move_to],true,false);
        
    }
  
  
  
}

void Verify(vector<HardBlock> &hb)
{
    for (int i = 0; i < num_hardblocks; i++) {
        int x_bl1 = hb[i].x;
        int y_bl1 = hb[i].y;
        int x_tr1 = x_bl1 + hb[i].width;
        int y_tr1 = y_bl1 + hb[i].height;
        for (int j = i + 1; j < num_hardblocks; j++) {
            //if (i == j)
            //    continue;

            int x_bl2 = hb[j].x;
            int y_bl2 = hb[j].y;
            int x_tr2 = x_bl2 + hb[j].width;
            int y_tr2 = y_bl2 + hb[j].height;

            if (!(x_tr1 <= x_bl2 || x_bl1 >= x_tr2 || y_tr1 <= y_bl2 || y_bl1 >= y_tr2)) {
                //printf("[Error] hardblocks overlapped\n");
                exit(1);
            }
            else {
                //printf("No hardblocks overlapped: \n");
                //cout << 'Blk:' <<  i << j << '\n';
                //cout << 'Pos: ' << x_bl1 << y_bl1 << x_tr1 << y_tr1 << x_bl2 << y_bl2 << x_tr2 << y_tr2 << '\n'; 
            }
        }

    }
}


void printboundary() {
    //cout << "***********************************************" << "\n";
    //cout << "Printing current boundary blocks from the layout " << "\n";
    cout << "Corner constraints " << "\n";
    cout << "SW: " << graph_sw << " SE: " << graph_se << " NW: " << graph_nw << " NE: " << graph_ne << "\n";

    cout << "South Region:" << "\n";
    for (int element : southG) {
        cout << element << " ";
    }
    cout << "\n";

    cout << "North Region:" << "\n";
    for (int element : northG) {
        cout << element << " ";
    }
    cout << "\n";

    cout << "East Region:" << "\n";
    for (int element : eastG) {
        cout << element << " ";
    }
    cout << "\n";

    cout << "West Region:" << "\n";
    for (int element : westG) {
        cout << element << " ";
    }
    cout << "\n";
}



void SimulatedAnnealing()
{
    //different sequence each time
    

    min_cost = CalculateCost();
    min_cost_floorplan = hardblocks;
    min_cost_btree = btree;
    min_cost_root_block = root_block;
    
    const double P = 0.95;
    const double r = 1.5;
    //const double epsilon = 0.001; // coolest temperature
    const int k = 10;//reduced from 20
    const int N = k * num_hardblocks;
    //Initial temperature (0.01) start with low temperature, and increase
    cout << "MIN COST = "<< min_cost.cost << "\n";
    double T = T0 ;
    cout << "INITIAL TEMPERATURE = " <<  T0 << "\n";

    int MT = 0;
    int uphill = 0;
    int reject = 0;
    Cost prev_cost = min_cost;
    in_fixed_outline = false;

    clock_t init_time = clock();
    clock_t time = init_time;
    const int max_seconds = tlimit;//in seconds
    const int TIME_LIMIT = tlimit;//in seconds
    const int STEP_LIMIT = step_limit;//count = 3000
    const int TEMP_COUNTER = step_limit / 10;//count = 300
    //temp points = step_limit / 10 = 300 steps for each temperature


    int seconds = 0, runtime = 0;
    int step_loop, temp_loop;
    step_loop = 0;

    location_trajectory.clear();
    state_trajectory.clear();
    action_trajectory.clear();
    reward_trajectory.clear();

    //initialize l, s, a, r
    std::vector<int> init_action;
    init_action.push_back(-1);
    init_action.push_back(-1);
    init_action.push_back(-1);
    action_trajectory.push_back(init_action);

    location_trajectory.push_back(current_loc);

    state_trajectory.push_back(current_btree);

    std::vector<double> init_reward;
    init_reward.push_back(min_cost.ws);
    init_reward.push_back(min_cost.wirelength);
    init_reward.push_back(min_cost.cost);
    init_reward.push_back(min_cost.n_violated_edge_constraints);
    init_reward.push_back(min_cost.clustering_cost);    
    reward_trajectory.push_back(init_reward);

    do {
        MT = 0;
        uphill = 0;
        reject = 0;
        temp_loop = 0;
        do {
            vector<HardBlock> hardblocks_temp(hardblocks);
            vector<Node> btree_temp(btree);
            int prev_root_block = root_block;

            std::vector<int> current_action;
            std::vector<double> current_reward;

            int M;
            if(ar_search)
              M = 0;
            else
              M = (rand() % 2) + 1;//commented AR search and rotate


            if (M == 0) {
                // AR search
              int node;
              do {
                node = rand() % num_hardblocks;
              }while (hardblocks[node].fixed_block);
                ARAdjust(node);
                current_action.push_back(node);
                current_action.push_back(-1);
                current_action.push_back(0);
            }
            else if (M == 1) {//add checks for boundary constraint
                // swap
                int node1, node2;
                do {
                    node1 = rand() % num_hardblocks;
                    node2 = rand() % num_hardblocks;
                } while (node2 == node1);
                Swap(node1, node2);
                current_action.push_back(node1);
                current_action.push_back(node2);
                current_action.push_back(1);
            }
            else if (M == 2) {//add checks for boundary constraint
                // move
                int node, to_node;
                int insert_flag = 0;
                int delete_flag = 0;
                do {
                    node = rand() % num_hardblocks;
                    to_node = rand() % num_hardblocks;
                } while (to_node == node ||  btree[to_node].parent == node);
                //if node and to_node have parent-child relation, move is unncessary.|| btree[node].parent == to_node
                Move(node, to_node);
                current_action.push_back(node);
                current_action.push_back(to_node);
                current_action.push_back(2);
            }
            else {
                cout << "[Error] Unspecified Move\n";
                exit(1);
            }

            MT++;
            Cost cur_cost = CalculateCost();//Tree to Floorplan conversion done
            double delta_cost = cur_cost.cost - prev_cost.cost;//negative is good
            double random = ((double)rand()) / RAND_MAX;
            double delta_ws = abs(cur_cost.ws - prev_cost.ws);
            bool ws_flag = true;//default
            if (inverse && delta_ws > ws_thresh) {
                ws_flag = false;
            } 
            //std::cout << "DELTA WS = " << delta_ws << "::" << ws_flag << "\n";
            if ( ( delta_cost < 0 || random < exp(-delta_cost / T) ) && ws_flag  ){//accept
                if (delta_cost > 0)
                    //cout << "Going uphill ....." << "\n";
                    uphill++;//cost is increasing

                // feasible solution found within the fixed-outline


                // infeasible solution with minimum cost
                if (cur_cost.cost < min_cost.cost) {
                    min_cost_root_block = root_block;
                    min_cost = cur_cost;
                    min_cost_floorplan = hardblocks;
                    min_cost_btree = btree;
                }

                prev_cost = cur_cost;
                current_reward.push_back(cur_cost.ws);
                current_reward.push_back(cur_cost.wirelength);
                current_reward.push_back(cur_cost.cost);
                current_reward.push_back(cur_cost.n_violated_edge_constraints);
                current_reward.push_back(cur_cost.clustering_cost);                    
                //Collect trajectory info

                //std:cout << "Random action: " << current_action[0] << "::" << current_action[1] << "::" << current_action[2] << "\n";

                location_trajectory.push_back(current_loc);
                state_trajectory.push_back(current_btree);
                action_trajectory.push_back(current_action);
                reward_trajectory.push_back(current_reward);

                if (step_loop % 1 == 0) {
                    wspace_list.push_back(cur_cost.ws);
                    wl_list.push_back(cur_cost.wirelength);
                    cost_list.push_back(cur_cost.cost);
                    //block_state_list.push_back(hardblocks);
                    //btree_state_list.push_back(btree);
                    //action_list.push_back(action);
                }
            }
            else {
                reject++;
                root_block = prev_root_block;
                if (M == 0)
                    hardblocks = hardblocks_temp;
                else
                    btree = btree_temp;
            }
            temp_loop++;//only succesfful actions
            step_loop++;//KEEP increasing but do not reset after every temperature change
            
            random = ((double)rand()) / RAND_MAX;
            if (random < constraint_fixing_prob)
              {
                //py::print("fixing constraint violations",random);
                BtreeToFloorplan();
                vector<int> violating_nodes;                
                for (int i = 0; i < num_hardblocks; i++)
                  {
                    if(hardblocks[i].required_loc != NC && hardblocks[i].required_loc !=  hardblocks[i].actual_loc)
                      violating_nodes.push_back(i);
                  }
                if(violating_nodes.size() > 0)
                  {
                    int node_to_fix = rand() % violating_nodes.size();
                    fix_edge_violation(violating_nodes[node_to_fix]);
                    prev_cost = CalculateCost();
                  }
                
              }
            
        } while (temp_loop < TEMP_COUNTER);//inner-loop ends after TEMP_COUNTER successfuls steps
        //        T *= r;
        

        //cout << "seconds and runtime " << seconds << " " << runtime << '\n';
    } while (step_loop < STEP_LIMIT);//outer-loop ends
    


}

void OutputFloorplan(string output_file, int wirelength, vector<HardBlock> &hb)
{
    ofstream file;
    file.open(output_file);

    file << "Wirelength " << wirelength << '\n';
    file << "Blocks\n";

    for (int i = 0; i < num_hardblocks; i++) {
        if (hb[i].rotate)
            file << "sb" << i << " " << hb[i].x << " " << hb[i].y << " " << hb[i].height << " " << hb[i].width << " 1\n";
        else
            file << "sb" << i << " " << hb[i].x << " " << hb[i].y << " " << hb[i].width << " " << hb[i].height << " 0\n";
    }

    file.close();
}

unsigned int GetRandomSeed()
{
    if (num_hardblocks == 100) {
        if (white_space_ratio == 0.1)
            return 1542894266;
        else if (white_space_ratio == 0.15)
            return 1542894588;
    }
    else if (num_hardblocks == 200) {
        if (white_space_ratio == 0.1)
            return 1542892927;
        else if (white_space_ratio == 0.15)
            return 1542892927;
    }
    else if (num_hardblocks == 300) {
        if (white_space_ratio == 0.1)
            return 1542959801;
        else if (white_space_ratio == 0.15)
            return 1542955417;
    }

    return time(NULL);
}

void writeMetrics() 
{
    //metric writing (WS, WL and Cost metrics)
    std::ofstream outputFileWS("ws_"+std::to_string(alpha)+"_"+std::to_string(_beta)+"_"+std::to_string(tlimit)+".txt");
    std::copy(wspace_list.begin(), wspace_list.end(), std::ostream_iterator<double>(outputFileWS, "\n"));
    outputFileWS.close();

    std::ofstream outputFileWL("wire_"+std::to_string(alpha)+"_"+std::to_string(_beta)+"_"+std::to_string(tlimit)+".txt");
    std::copy(wl_list.begin(), wl_list.end(), std::ostream_iterator<double>(outputFileWL, "\n"));
    outputFileWL.close();

    std::ofstream outputFileCost("cost_"+std::to_string(alpha)+"_"+std::to_string(_beta)+"_"+std::to_string(tlimit)+".txt");
    std::copy(cost_list.begin(), cost_list.end(), std::ostream_iterator<double>(outputFileCost, "\n"));
    outputFileCost.close();

}


// Define a new C++ function that you want to expose to Python

//get and set btree
void init_btree(){
    //std::vector<std::vector<int>> current_btree;
    current_btree.clear();
}

std::vector<std::vector<std::vector<int>>> get_location_trajectory(){
    return location_trajectory;
}

std::vector<std::vector<std::vector<int>>> get_state_trajectory(){
    return state_trajectory;
}

void clear_trajectory(){
    location_trajectory.clear();
    state_trajectory.clear();
    action_trajectory.clear();
    reward_trajectory.clear();
}
std::vector<std::vector<int>> get_action_trajectory(){
    return action_trajectory;
}

std::vector<std::vector<double>> get_reward_trajectory(){
    return reward_trajectory;
}

std::vector<std::vector<int>> get_btree(){
    return current_btree;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}
void set_btree(std::vector<std::vector<int>> tree_list){
    BuildPreExistBTree(tree_list);
    BtreeToFloorplan();// Convert tree to layout, to estimate cost
}

//hard actions (without checking cost)
void swap_nodes(int n1, int n2) {
    Swap(n1, n2);
}
void move_nodes(int n1, int n2) {
    Move(n1, n2);
}
void rotate_node(int n1) {
    Rotate(n1);
}


//compute cost using CalculateCost();

// Function to get the value of chip_ar
double get_chip_ar() {
    return chip_ar;
}

// Function to set the value of chip_ar
void set_chip_ar(double value) {
    chip_ar = value;
}


// Function to get the value of delta_ws thresh %
double get_ws_thresh() {
    return ws_thresh;
}

// Function to set the value of delta_ws thresh %
void set_ws_thresh(double value) {
    ws_thresh = value;
}

// Function to get the value of inverse flag
double get_inverse() {
    return inverse;
}

// Function to set the value of inverse flag
void set_inverse(bool value) {
    inverse = value;
}


// Function to get the value of alpha
double get_alpha() {
    return alpha;
}

// Function to set the value of alpha
void set_alpha(double value) {
    alpha = value;
}



// Function to get the value of alpha
double get_ar_increment() {
    return ar_increment;
}

// Function to set the value of alpha
void set_ar_increment(double value) {
    ar_increment = value;
}



// Function to get the value of alpha
double get_beta() {
    return _beta;
}

// Function to set the value of alpha
void set_beta(double value) {
    _beta = value;
}



// Function to get the value of alpha
double get_beta_cluster() {
    return beta_cluster;
}

// Function to set the value of alpha
void set_beta_cluster(double value) {
    beta_cluster = value;
}


// Function to get the value of alpha
double get_beta_rectilinear() {
    return beta_rectilinear;
}

// Function to set the value of alpha
void set_beta_rectilinear(double value) {
    beta_rectilinear = value;
}


// Function to get the value of alpha
double get_beta_preplaced() {
    return beta_preplaced;
}

// Function to set the value of alpha
void set_beta_preplaced(double value) {
    beta_preplaced = value;
}



// Function to get the value of alpha
double get_fixing_prob() {
    return constraint_fixing_prob;
}

// Function to set the value of alpha
void set_fixing_prob(double value) {
    constraint_fixing_prob = value;
}



// Function to get the value of alpha
double get_T0() {
    return T0;
}

// Function to set the value of alpha
void set_T0(double value) {
    T0 = value;
}


// Function to get the value of step_limit
double get_step_limit() {
    return step_limit;
}

// Function to set the value of step_limit
void set_step_limit(double value) {
    step_limit = value;
}


// Function to set the value of step_limit
void set_ar_search(bool value) {
    ar_search = value;
}

// Function to get the value of white_space_ratio
double get_ws_ratio() {
    return white_space_ratio;
}

// Function to set the value of step_limit
void set_ws_ratio(double value) {
    white_space_ratio = value;
}

void set_floorplan_boundaries(int width,int height,double weight,bool enable)
{
  enable_floorplan_boundaries = enable;
  floorplan_width = width;
  floorplan_height = height;
  floorplan_boundaries_weight = weight;
}
    

//Function to compute cost and return ws, wirelength, area and total cost
std::vector<double> compute_cost() {
    Cost qc = CalculateCost();
    std::vector<double> result;
    result.push_back(qc.ws);
    result.push_back(qc.wirelength);
    result.push_back(qc.area);
    result.push_back(qc.n_violated_edge_constraints);
    result.push_back(qc.extra_cluster_fragments);
    result.push_back(qc.extra_rectilinear_fragments);            
    result.push_back(qc.cost);
    result.push_back(qc.width_penalty);
    result.push_back(qc.height_penalty);    
    result.push_back(qc.preplaced_cost);
    
    return result;
}




//Function to initialize default configurations
// chip_ar, white_space_ratio, alpha, and step_limit
void initialize(int random_tree){
    //cout << "***Reading input files: " << hardblocks_file << "," << nets_file << "," << terminals_file << "\n";
    //ReadHardblocksFile(hardblocks_file);
    //ReadNetsFile(nets_file);
    //ReadTerminalsFile(terminals_file);

    //cout << "***Checking for the loaded hardblock list/vector \n";
    //for (int i = 0; i < num_hardblocks; i++) {
    //    cout << i << "::" << hardblocks[i].x << ":" << hardblocks[i].width << "--" << hardblocks[i].y << ":" << hardblocks[i].height << "\n";
    //}

    // cout << "***Checking for the loaded nets list/vector \n";
    // for (const vector<int> &net : nets) {
    //     cout << "NET is connected to the following blocks: " << "\n";
    //     for (const int pin : net) {
    //         cout << " Block: " << pin ;
    //     }
    //     cout << "\n";
    // }

    //cout << "***Checking for the loaded terminals list/vector \n";
    //cout << "Number of terminals = " << num_terminals << "\n";
    // for (int i = 1; i <= num_terminals; i++) {
    //     cout << "Terminal: " << i << ":" << terminals[i].x << " :: " << terminals[i].y ;
    // }

    // cout << "***Checking for global variable assignment: \n";
    // cout << "alpha = " << alpha << " chip_ar = " << chip_ar << " step limit = " << step_limit << " ws = " <<  white_space_ratio << "\n";
    // // Start with a random tree
    // cout << "***Starting to build initial tree " << "\n";
    if (random_tree) {//else, expect the tree to be initialied using load_btree
        BuildInitBtree();//can be changed using set_btree
    }
    BtreeToFloorplan();// Convert tree to layout, to estimate cost
}

void sa_refine() {
    BtreeToFloorplan();// Convert tree to layout, to estimate cost
    //cout << "***Calling Simulated Annealing\n";
    SimulatedAnnealing();//Run SA 
    //writeMetrics();//write cost reports
}

void go_to_min_cost_sol()
{
  hardblocks = min_cost_floorplan;
  btree = min_cost_btree;
  root_block = min_cost_root_block;
}
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize, "Simulated Annealing Refinement");
        //  py::arg("hardblocks_file"),
        //  py::arg("nets_file"),
        //  py::arg("terminals_file"));
        //   py::arg("chip_ar") = 1.0,  // Default values for additional parameters
        //   py::arg("white_space_ratio") = 0.2,
        //   py::arg("alpha") = 0.5,
        //   py::arg("step_limit") = 300.0);  

    m.def("init_btree", &init_btree, "Initialize btree vector");
    m.def("go_to_min_cost_sol", &go_to_min_cost_sol, "Activate min cost solution");    
    m.def("sa_refine", &sa_refine, "Perform refinement using SA search");

    m.def("get_chip_ar", &get_chip_ar, "Get the value of chip_ar");
    m.def("set_chip_ar", &set_chip_ar, "Set the value of chip_ar", py::arg("value"));

    m.def("get_T0", &get_T0, "Get the initial temperature");
    m.def("set_T0", &set_T0, "Set the initial temperature", py::arg("value"));
    
    m.def("get_ws_thresh", &get_ws_thresh, "Get the value of ws thresh percent");
    m.def("set_ws_thresh", &set_ws_thresh, "Set the value of ws thresh percent", py::arg("value"));

    m.def("get_inverse", &get_inverse, "Get the value of inverse flag");
    m.def("set_inverse", &set_inverse, "Set the value of inverse flag", py::arg("value"));

    m.def("get_alpha", &get_alpha, "Get the value of alpha");
    m.def("set_alpha", &set_alpha, "Set the value of alpha", py::arg("value"));

    m.def("get_ar_increment", &get_ar_increment, "Get the value of AR increment");
    m.def("set_ar_increment", &set_ar_increment, "Set the value of AR increment", py::arg("value"));
    

    m.def("get_fixing_prob", &get_fixing_prob, "Get constraint fixing probability");
    m.def("set_fixing_prob", &set_fixing_prob, "Set the constraint fixing probability", py::arg("value"));
    
    m.def("get_beta", &get_beta, "Get the value of beta");
    m.def("set_beta", &set_beta, "Set the value of beta", py::arg("value"));
 
    m.def("get_beta_cluster", &get_beta_cluster, "Get the value of beta");
    m.def("set_beta_cluster", &set_beta_cluster, "Set the value of beta", py::arg("value"));

    m.def("get_beta_rectilinear", &get_beta_rectilinear, "Get the value of beta");
    m.def("set_beta_rectilinear", &set_beta_rectilinear, "Set the value of beta", py::arg("value"));


    m.def("get_beta_preplaced", &get_beta_preplaced, "Get the value of beta preplaced");
    m.def("set_beta_preplaced", &set_beta_preplaced, "Set the value of beta preplaced", py::arg("value"));
    
    
    m.def("get_step_limit", &get_step_limit, "Get the value of step_limit");
    m.def("set_step_limit", &set_step_limit, "Set the value of step_limit", py::arg("value"));

    m.def("ar_search", &set_ar_search, "Enable or disable AR search", py::arg("value"));

    
    m.def("get_ws_ratio", &get_ws_ratio, "Get the value of ws ratio");
    m.def("set_ws_ratio", &set_ws_ratio, "Set the value of ws ratio", py::arg("value"));

    m.def("set_floorplan_boundaries", &set_floorplan_boundaries, "Set the value of ws ratio",
          py::arg("value"),py::arg("value"),py::arg("value"),py::arg("value"));    

    m.def("get_btree", &get_btree, "Get the btree");
    m.def("get_bpos", &getBlockPos, "Get the block positions and dimensions");
    m.def("get_problem", &getProblem, "Get the block positions and dimensions");    
    m.def("getClusterdata", &getClusterData, "Get the clustering data",py::arg("value"));    
    m.def("set_btree", &set_btree, "Set a btree using a tree file", py::arg("value"));

    m.def("set_seed", &set_seed, "Set a seed", py::arg("value"));    

    m.def("swap_nodes", &swap_nodes, "Swap two nodes n1 and n2");    
    m.def("move_nodes", &move_nodes, "Move node from n1 to n2");
    m.def("ar_adjust", &ARAdjust, "AR adjust");

    m.def("hard_preplace_constraints", &hard_preplace_constraints, "AR adjust");
    
    m.def("rotate_node", &rotate_node, "Rotate a node n1");
    m.def("fix_edge_violation", &fix_edge_violation, "Fix an edge violation");    

    m.def("compute_cost", &compute_cost, "Compute cost for the current btree");

    m.def("get_location_trajectory",   &get_location_trajectory,  "returns 3-D location vector");
    m.def("get_state_trajectory",   &get_state_trajectory,  "returns 3-D state vector");
    m.def("get_action_trajectory",  &get_action_trajectory, "returns 2-D action vector");
    m.def("get_reward_trajectory",  &get_reward_trajectory, "returns 2-D reward vector");


    m.def("read_blocks", &ReadHardBlocks, "To read and load the block dimensions", py::arg("value"),py::arg("value"));
    m.def("read_nets", &ReadNets, "To read and load the net connectivity", py::arg("value"));
    m.def("read_terminals", &ReadTerminals, "To read and load the terminal locations", py::arg("value"));


    m.def("load_btree", &LoadBTree, "To read and load an existing btree", py::arg("value"));

    m.def("clear_trajectory",  &clear_trajectory, "clears the trajectory data");
    m.attr("__version__") = "0.0.5";
}
