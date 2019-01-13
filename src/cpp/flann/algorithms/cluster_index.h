/**
独自クラス
**/


#ifndef FLANN_CLUSTER_INDEX_H_
#define FLANN_CLUSTER_INDEX_H_

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>
#include <iostream>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include <flann/algorithms/center_chooser.h>
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"
#include "flann/util/logger.h"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cstdlib>

namespace flann
{

  struct ClusterIndexParams : public IndexParams
  {
    ClusterIndexParams(std::string tree_path = std::string("tree.json"), float cb_index = 0.2)
    {
      (* this)["algorithm"] = FLANN_INDEX_CLUSTER;
      //外部ツリーパス
      (* this)["tree_path"] = tree_path;
      // cluster boundary index. Used when searching the cluster tree
      (*this)["cb_index"] = cb_index;
    }
  };

  /**
  * Cluster index
  */
  template <typename Distance>
  class ClusterIndex : public NNIndex<Distance>
  {
  public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    flann_algorithm_t getType() const
    {
      return FLANN_INDEX_CLUSTER;
    }

    /**
    * Index constructor
    *
    * Params:
    *          inputData = dataset with the input features
    *          params = parameters passed to the cluster algorithm
    */
    ClusterIndex(const Matrix<ElementType>& input_data, const IndexParams& params = ClusterIndexParams(), Distance d = Distance()) :
    BaseClass(params, d), root_(NULL), memoryCounter_(0)
    {
      tree_path_ = get_param(params,"tree_path",std::string(""));
      cb_index_  = get_param(params,"cb_index",0.4f);
      setDataset(input_data);
    }

    /**
    * Index constructor
    *
    * Params:
    *          params = parameters passed to the cluster algorithm
    */
    ClusterIndex(const IndexParams& params = ClusterIndexParams(), Distance d = Distance()) :
    BaseClass(params, d), root_(NULL), memoryCounter_(0)
    {
      tree_path_ = get_param(params,"tree_path", std::string("tree.json"));
      cb_index_  = get_param(params,"cb_index",0.4f);
    }

    ClusterIndex(const ClusterIndex& other) : BaseClass(other),
    tree_path_(other.tree_path_),
    memoryCounter_(other.memoryCounter_)
    {
      copyTree(root_, other.root_);
    }

    ClusterIndex& operator=(ClusterIndex other)
    {
      this->swap(other);
      return *this;
    }

    virtual ~ClusterIndex()
    {
      freeIndex();
    }

    BaseClass* clone() const
    {
      return new ClusterIndex(*this);
    }

    /**
    * Computes the inde memory usage
    * Returns: memory used by the index
    */
    int usedMemory() const
    {
      return pool_.usedMemory+pool_.wastedMemory+memoryCounter_;
    }

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
      std::cerr << "Can not add Points to this index" << std::endl;
      exit(1);
    }

    template<typename Archive>
    void serialize(Archive& ar)
    {
      ar.setObject(this);

      ar & *static_cast<NNIndex<Distance>*>(this);

      ar & memoryCounter_;
      ar & cb_index_;

      if (Archive::is_loading::value) {
        root_ = new(pool_) Node();
      }
      ar & *root_;

      if (Archive::is_loading::value) {
        index_params_["algorithm"] = getType();
        index_params_["tree_path"] = tree_path_;
        index_params_["cb_index"] = cb_index_;
      }
    }

    void saveIndex(FILE* stream)
    {
      serialization::SaveArchive sa(stream);
      sa & *this;
    }

    void loadIndex(FILE* stream)
    {
      freeIndex();
      serialization::LoadArchive la(stream);
      la & *this;
    }

    /**
    * Find set of nearest neighbors to vec. Their indices are stored inside
    * the result object.
    *
    * Params:
    *     result = the result object in which the indices of the nearest-neighbors are stored
    *     vec = the vector for which to search the nearest neighbors
    *     searchParams = parameters that influence the search algorithm (checks, cb_index)
    */

    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
      if (removed_) {
        findNeighborsWithRemoved<true>(result, vec, searchParams);
      }
      else {
        findNeighborsWithRemoved<false>(result, vec, searchParams);
      }

    }


  protected:
    /**
    * Builds the index
    */
    void buildIndexImpl()
    {
      std::vector<int> indices(size_);
      for (size_t i=0; i<size_; ++i) {
        indices[i] = int(i);
      }

      root_ = new(pool_) Node();
      computeNodeStatistics(root_, indices);

      std::string tree_file_path = std::string(tree_path_);

      boost::property_tree::ptree root;
      boost::property_tree::read_json(tree_file_path, root);

      std::cout << "Tree Path: " << tree_file_path << std::endl;

      parseTree(root_, root);
    }

  private:
    struct PointInfo
    {
      size_t index;
      ElementType* point;
    private:
      template<typename Archive>
      void serialize(Archive& ar)
      {
        typedef ClusterIndex<Distance> Index;
        Index* obj = static_cast<Index*>(ar.getObject());

        ar & index;
        //    		ar & point;

        if (Archive::is_loading::value) point = obj->points_[index];
      }
      friend struct serialization::access;
    };

    /**
    * Struture representing a node in the hierarchical k-means tree.
    */
    struct Node
    {
      /**
      * The cluster center.
      */
      DistanceType* pivot;
      /**
      * The cluster radius.
      */
      DistanceType radius;
      /**
      * The cluster variance.
      */
      DistanceType variance;
      /**
      * The cluster size (number of points in the cluster)
      */
      int size;
      /**
      * Child nodes (only for non-terminal nodes)
      */
      std::vector<Node*> childs;
      /**
      * Node points (only for terminal nodes)
      */
      std::vector<PointInfo> points;
      /**
      * Level
      */
      //        int level;

      ~Node()
      {
        delete[] pivot;
        if (!childs.empty()) {
          for (size_t i=0; i<childs.size(); ++i) {
            childs[i]->~Node();
          }
        }
      }

      template<typename Archive>
      void serialize(Archive& ar)
      {
        typedef ClusterIndex<Distance> Index;
        Index* obj = static_cast<Index*>(ar.getObject());

        if (Archive::is_loading::value) {
          delete[] pivot;
          pivot = new DistanceType[obj->veclen_];
        }
        ar & serialization::make_binary_object(pivot, obj->veclen_*sizeof(DistanceType));
        ar & radius;
        ar & variance;
        ar & size;

        size_t childs_size;
        if (Archive::is_saving::value) {
          childs_size = childs.size();
        }
        ar & childs_size;

        if (childs_size==0) {
          ar & points;
        }
        else {
          if (Archive::is_loading::value) {
            childs.resize(childs_size);
          }
          for (size_t i=0;i<childs_size;++i) {
            if (Archive::is_loading::value) {
              childs[i] = new(obj->pool_) Node();
            }
            ar & *childs[i];
          }
        }
      }
      friend struct serialization::access;
    };
    typedef Node* NodePtr;

    /**
     * Alias definition for a nicer syntax.
     */
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;

    /**
    * Helper function
    */
    void freeIndex()
    {
      if (root_) root_->~Node();
      root_ = NULL;
      pool_.free();
    }

    void copyTree(NodePtr& dst, const NodePtr& src)
    {
      dst = new(pool_) Node();
      dst->pivot = new DistanceType[veclen_];
      std::copy(src->pivot, src->pivot+veclen_, dst->pivot);
      dst->radius = src->radius;
      dst->variance = src->variance;
      dst->size = src->size;

      if (src->childs.size()==0) {
        dst->points = src->points;
      }
      else {
        dst->childs.resize(src->childs.size());
        for (size_t i=0;i<src->childs.size();++i) {
          copyTree(dst->childs[i], src->childs[i]);
        }
      }
    }

    /**
    * Computes the statistics of a node (mean, radius, variance).
    *
    * Params:
    *     node = the node to use
    *     indices = the indices of the points belonging to the node
    */
    void computeNodeStatistics(NodePtr node, const std::vector<int>& indices)
    {
      size_t size = indices.size();

      DistanceType* mean = new DistanceType[veclen_];
      memoryCounter_ += int(veclen_*sizeof(DistanceType));
      memset(mean,0,veclen_*sizeof(DistanceType));

      for (size_t i=0; i<size; ++i) {
        ElementType* vec = points_[indices[i]];
        for (size_t j=0; j<veclen_; ++j) {
          mean[j] += vec[j];
        }
      }
      DistanceType div_factor = DistanceType(1)/size;
      for (size_t j=0; j<veclen_; ++j) {
        mean[j] *= div_factor;
      }

      DistanceType radius = 0;
      DistanceType variance = 0;
      for (size_t i=0; i<size; ++i) {
        DistanceType dist = distance_(mean, points_[indices[i]], veclen_);
        if (dist>radius) {
          radius = dist;
        }
        variance += dist;
      }
      variance /= size;

      node->variance = variance;
      node->radius = radius;
      delete[] node->pivot;
      node->pivot = mean;
    }

    /**
    * 自作関数
    */
    std::vector<int> getLeafs(NodePtr node){
      std::vector<int> result;

      for(auto elem: node->points){
        result.push_back(elem.index);
      }

      for(auto child: node->childs){
        std::vector<int> child_points = getLeafs(child);
        for(auto elem: child_points){
          result.push_back(elem);
        }
      }

      return result;
    }

    /**
    * 自作関数
    */
    NodePtr parseTree(NodePtr node, const boost::property_tree::ptree &json_node){
      //points
      for(auto &clusters: json_node.get_child("points")){
        //葉ノードを作成
        struct Node *child_node = new(pool_) Node();

        child_node->points.resize(clusters.second.size());
        int count = 0;

        for(auto &points : clusters.second){
          int index = points.second.get_value<int>();
          child_node->points[count].index = index;
          child_node->points[count].point = points_[index];
          count++;
        }
        child_node->childs.clear();

        auto indices = getLeafs(child_node);
        child_node->size = indices.size();

        computeNodeStatistics(child_node, indices);
        //節ノードに葉ノードを追加
        node->childs.push_back(child_node);
      }

      //childs
      BOOST_FOREACH(const boost::property_tree::ptree::value_type& child, json_node.get_child("childs")){
        const boost::property_tree::ptree& childs = child.second;

        struct Node *child_node = new(pool_) Node();
        parseTree(child_node, childs);

        auto indices = getLeafs(child_node);
        child_node->size = indices.size();

        computeNodeStatistics(child_node, indices);

        //節ノードに節ノードを追加
        node->childs.push_back(child_node);
      }

      auto indices = getLeafs(node);
      node->size = indices.size();

      computeNodeStatistics(node, indices);

      return node;
    }

    template<bool with_removed>
    void findNeighborsWithRemoved(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {

      int maxChecks = searchParams.checks;

      if (maxChecks==FLANN_CHECKS_UNLIMITED) {
        findExactNN<with_removed>(root_, result, vec);
      }
      else {
        // Priority queue storing intermediate branches in the best-bin-first search
        Heap<BranchSt>* heap = new Heap<BranchSt>((int)size_);

        int checks = 0;
        findNN<with_removed>(root_, result, vec, checks, maxChecks, heap);

        BranchSt branch;
        while (heap->popMin(branch) && (checks<maxChecks || !result.full())) {
          NodePtr node = branch.node;
          findNN<with_removed>(node, result, vec, checks, maxChecks, heap);
        }

        delete heap;
      }

    }

    template<bool with_removed>
    void findNN(NodePtr node, ResultSet<DistanceType>& result, const ElementType* vec, int& checks, int maxChecks,
      Heap<BranchSt>* heap) const
      {
        // Ignore those clusters that are too far away
        {
          DistanceType bsq = distance_(vec, node->pivot, veclen_);
          DistanceType rsq = node->radius;
          DistanceType wsq = result.worstDist();

          DistanceType val = bsq-rsq-wsq;
          DistanceType val2 = val*val-4*rsq*wsq;

          //if (val>0) {
          if ((val>0)&&(val2>0)) {
            return;
          }
        }

        if (node->childs.empty()) {
          if (checks>=maxChecks) {
            if (result.full()) return;
          }
          for (int i=0; i<node->size; ++i) {
            PointInfo& point_info = node->points[i];
            int index = point_info.index;
            if (with_removed) {
              if (removed_points_.test(index)) continue;
            }
            DistanceType dist = distance_(point_info.point, vec, veclen_);
            result.addPoint(dist, index);
            ++checks;
          }
        }
        else {
          int closest_center = exploreNodeBranches(node, vec, heap);
          findNN<with_removed>(node->childs[closest_center],result,vec, checks, maxChecks, heap);
        }
      }

      /**
      * Helper function that computes the nearest childs of a node to a given query point.
      * Params:
      *     node = the node
      *     q = the query point
      *     distances = array with the distances to each child node.
      * Returns:
      */
      int exploreNodeBranches(NodePtr node, const ElementType* q, Heap<BranchSt>* heap) const
      {
        int n_cluster = node->childs.size();

        std::vector<DistanceType> domain_distances(n_cluster);
        int best_index = 0;
        domain_distances[best_index] = distance_(q, node->childs[best_index]->pivot, veclen_);
        for (int i=1; i<n_cluster; ++i) {
          domain_distances[i] = distance_(q, node->childs[i]->pivot, veclen_);
          if (domain_distances[i]<domain_distances[best_index]) {
            best_index = i;
          }
        }

        //		float* best_center = node->childs[best_index]->pivot;
        for (int i=0; i<n_cluster; ++i) {
          if (i != best_index) {
            domain_distances[i] -= cb_index_*node->childs[i]->variance;

            //				float dist_to_border = getDistanceToBorder(node.childs[i].pivot,best_center,q);
            //				if (domain_distances[i]<dist_to_border) {
            //					domain_distances[i] = dist_to_border;
            //				}
            heap->insert(BranchSt(node->childs[i],domain_distances[i]));
          }
        }

        return best_index;
      }


      /**
      * Function the performs exact nearest neighbor search by traversing the entire tree.
      */
      template<bool with_removed>
      void findExactNN(NodePtr node, ResultSet<DistanceType>& result, const ElementType* vec) const
      {
        // Ignore those clusters that are too far away
        {
          DistanceType bsq = distance_(vec, node->pivot, veclen_);
          DistanceType rsq = node->radius;
          DistanceType wsq = result.worstDist();

          DistanceType val = bsq-rsq-wsq;
          DistanceType val2 = val*val-4*rsq*wsq;

          //                  if (val>0) {
          if ((val>0)&&(val2>0)) {
            return;
          }
        }

        int n_cluster = node->childs.size();

        if (node->childs.empty()) {
          for (int i=0; i<node->size; ++i) {
            PointInfo& point_info = node->points[i];
            int index = point_info.index;
            if (with_removed) {
              if (removed_points_.test(index)) continue;
            }
            DistanceType dist = distance_(point_info.point, vec, veclen_);
            result.addPoint(dist, index);
          }
        }
        else {
          std::vector<int> sort_indices(n_cluster);
          getCenterOrdering(node, vec, sort_indices);

          for (int i=0; i<n_cluster; ++i) {
            findExactNN<with_removed>(node->childs[sort_indices[i]],result,vec);
          }

        }
      }

      /**
      * Helper function.
      *
      * I computes the order in which to traverse the child nodes of a particular node.
      */
      void getCenterOrdering(NodePtr node, const ElementType* q, std::vector<int>& sort_indices) const
      {
        int n_cluster = node->childs.size();

        std::vector<DistanceType> domain_distances(n_cluster);
        for (int i=0; i<n_cluster; ++i) {
          DistanceType dist = distance_(q, node->childs[i]->pivot, veclen_);

          int j=0;
          while (domain_distances[j]<dist && j<i) j++;
          for (int k=i; k>j; --k) {
            domain_distances[k] = domain_distances[k-1];
            sort_indices[k] = sort_indices[k-1];
          }
          domain_distances[j] = dist;
          sort_indices[j] = i;
        }
      }

      /**
      * Method that computes the squared distance from the query point q
      * from inside region with center c to the border between this
      * region and the region with center p
      */
      DistanceType getDistanceToBorder(DistanceType* p, DistanceType* c, DistanceType* q) const
      {
        DistanceType sum = 0;
        DistanceType sum2 = 0;

        for (int i=0; i<veclen_; ++i) {
          DistanceType t = c[i]-p[i];
          sum += t*(q[i]-(c[i]+p[i])/2);
          sum2 += t*t;
        }

        return sum*sum/sum2;
      }

      void addPointToTree(NodePtr node, size_t index, DistanceType dist_to_pivot)
      {
        std::cerr << "Can not add Point to this tree" << std::endl;
        exit(1);
      }

      void swap(ClusterIndex& other)
      {
        std::swap(cb_index_, other.cb_index_);
        std::swap(root_, other.root_);
        std::swap(pool_, other.pool_);
        std::swap(memoryCounter_, other.memoryCounter_);
      }

    private:
      /** external tree path*/
      std::string tree_path_;

      /**
       * Cluster border index. This is used in the tree search phase when determining
       * the closest cluster to explore next. A zero value takes into account only
       * the cluster centres, a value greater then zero also take into account the size
       * of the cluster.
       */
      float cb_index_;

      /**
      * The root node in the tree.
      */
      NodePtr root_;

      /**
      * Pooled memory allocator.
      */
      PooledAllocator pool_;

      /**
      * Memory occupied by the index.
      */
      int memoryCounter_;

      USING_BASECLASS_SYMBOLS
    };

  }

  #endif // FLANN_Cluster_INDEX_H_
