#ifndef FAST_GICP_FAST_GICP_HPP
#define FAST_GICP_FAST_GICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/registration/registration.h>
#include <pcl/filters/filter.h>
#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>
#include <ctime>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */
template<typename PointSource, typename PointTarget, typename SearchMethodSource = pcl::search::KdTree<PointSource>, typename SearchMethodTarget = pcl::search::KdTree<PointTarget>>
class FastGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<FastGICP<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const FastGICP<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<FastGICP<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const FastGICP<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  // high_resolution points
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

  // edge points
  PointCloudSourcePtr input_edge_;
  PointCloudTargetPtr target_edge_;

  // low resolution
  PointCloudSourceConstPtr input_sparse_;

public:
  FastGICP();
  virtual ~FastGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);
  void setKNNMaxDistance(float k);
  void setEdgeWeight(float weight);
  void setSparseWeight(float weight);
  void setSourceCovMid(bool use_mid);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  void setInputSourceSparse(const PointCloudSourceConstPtr& cloud);
  void calculateSourceCovariance();
  virtual void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);
  virtual void setSourceCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales);
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  void calculateTargetCovariance();
  void calculateTargetCovarianceWithZ();
  void calculateTargetCovarianceWithFilter();
  virtual void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);
  virtual void setTargetCovariances(
    const std::vector<float>& input_rotationsq,
    const std::vector<float>& input_scales);
    
  void setTargetCovariancesAdditional(
    const std::vector<float>& input_rotationsq,
    const std::vector<float>& input_scales,
    const std::vector<int>& input_edge_mask);
  
  virtual void setSourceZvalues(const std::vector<float>& input_z_values);
  virtual void setTargetZvalues(const std::vector<float>& input_z_values);
  virtual void setSourceFilter(const int num_trackable_points, const std::vector<int>& input_filter);
  virtual void setTargetFilter(const int num_trackable_points, const std::vector<int>& input_filter);

  virtual void setSourceEdgeInvalidMask(const std::vector<int>& input_mask) { 
    invalid_edge_filter_source_.clear();
    invalid_edge_filter_source_ = input_mask;
  };

  virtual void setTargetEdgeInvalidMask(const std::vector<int>& input_mask) { 
    invalid_edge_filter_target_.clear();
    invalid_edge_filter_target_ = input_mask;
  };

  const std::vector<int>& getSourceCorrespondences() const { 
  	if (input_->size() != correspondences_.size()){ std::cerr<< "source and correspondence size mismatch. Did you change src after align()?"<<std::endl;}
  	return correspondences_; }
  const std::vector<float>& getSourceSqDistances() const {
  	if (input_->size() != sq_distances_.size()){ std::cerr<< "source and sq_distances size mismatch. Did you change src after align()?"<<std::endl;}
  	return sq_distances_;}

  const int getSourceSize() const {return input_->size();}
  const int getSourceRotationsqSize() const {return source_rotationsq_.size();}
  const int getSourceScaleSize() const {return source_scales_.size();}
  const int getSourceEdgeMaskSize() const {return is_edge_points_source_.size();}
  const int getTargetSize() const {return target_->size();}
  const int getTargetRotationsqSize() const {return target_rotationsq_.size();}
  const int getTargetScaleSize() const {return target_scales_.size();}
  const int getTargetEdgeMaskSize() const {return is_edge_points_target_.size();}

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getSourceCovariances() const {return source_covs_;}
  const std::vector<float>& getSourceRotationsq() const {
  	// if (input_->size() * 4 != source_rotationsq_.size()){ std::cerr << "source and quaternions size mismatch. Did you change source?"<<std::endl;}
  	return source_rotationsq_;}
  const std::vector<float>& getSourceScales() const {
	// if (input_->size() * 3 != source_scales_.size()){ std::cerr << "source and quaternions size mismatch. Did you change source?"<<std::endl;}
  	return source_scales_;}
  const std::vector<int>& getSourceEdgeMask() const {
    return is_edge_points_source_;
  }

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getTargetCovariances() const {return target_covs_;}
  const std::vector<float>& getTargetRotationsq() const {
  	// if (target_->size() * 4 != target_rotationsq_.size()){ std::cerr << "target and quaternions size mismatch. Did you change target?"<<std::endl;}
    return target_rotationsq_;}
  const std::vector<float>& getTargetScales() const {
  	// if (target_->size() * 3 != target_scales_.size()){ std::cerr << "target and quaternions size mismatch. Did you change target?"<<std::endl;}
    return target_scales_;}
  const std::vector<int>& getTargetEdgeMask() const {
    return is_edge_points_target_;
  }

  
  const int getSourceKnnIdxsSize() const { return source_knn_idxs_.size(); }
  const std::vector<int>& getSourceKnnIdxs() const { return source_knn_idxs_; }

  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

protected:
  // virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void update_correspondences(const Eigen::Isometry3d& trans);

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  template<typename PointT>
  bool calculate_covariances(
  	const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  	pcl::search::Search<PointT>& kdtree,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales,
  std::vector<int>& knn_idxs_);

  template<typename PointT>
  bool calculate_covariances_withz(
  	const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  	pcl::search::Search<PointT>& kdtree,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales,
  std::vector<float>& z_values);

  template<typename PointT>
  bool calculate_source_covariances_with_filter(
  	const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    PointCloudSourcePtr& edge_cloud,
  	pcl::search::Search<PointT>& kdtree,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales,
  std::vector<int>& filter,
  std::vector<int>& is_edge_points);

  template<typename PointT>
  bool calculate_target_covariances_with_filter(
  	const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    PointCloudTargetPtr& edge_cloud,
    pcl::search::Search<PointT>& edge_kdtree,
  	pcl::search::Search<PointT>& kdtree,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales,
  std::vector<int>& filter,
  std::vector<int>& is_edge_points);
  

  void setCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales);

  template<typename PointT>
  void setCovariancesAdditional(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  const std::vector<int>& input_edge_mask,
  PointCloudTargetPtr& edge_cloud,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
	std::vector<float>& rotationsq,
	std::vector<float>& scales);
  
protected:
  int num_threads_;
  int k_correspondences_;
  float knn_max_distance_;

  RegularizationMethod regularization_method_;

  std::shared_ptr<SearchMethodSource> search_source_;
  std::shared_ptr<SearchMethodSource> search_source_sparse_;
  std::shared_ptr<SearchMethodTarget> search_target_;
  std::shared_ptr<SearchMethodTarget> search_target_edge_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_;               // base resolution
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_sparse_covs_; // low resolution
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_edge_covs_;          // edge points
//  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> source_rotationsq_;
//  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_scales_;
  std::vector<float> source_rotationsq_;
  std::vector<float> source_scales_;
  std::vector<float> source_z_values_;
  std::vector<int> source_filter_;
  int source_num_trackable_points_;
  std::vector<int> source_knn_idxs_;
  std::vector<int> target_knn_idxs_;
  
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_edge_covs_;
//  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> target_rotationsq_;
//  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_scales_;
  std::vector<float> target_rotationsq_;
  std::vector<float> target_scales_;
  std::vector<float> target_z_values_;
  std::vector<int> target_filter_;
  int target_num_trackable_points_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis_;

  std::vector<int> correspondences_;
  std::vector<int> correspondences_extended_edge_; // for edge matching
  std::vector<int> correspondences_extended_sparse_;
  std::vector<float> sq_distances_;

  float matching_weight_edge_;
  float matching_weight_sparse_;
  bool source_cov_mid_;

  float edge_corr_dist_threshold_;

  std::vector<int> is_edge_points_source_;
  std::vector<int> is_edge_points_target_;

  std::vector<int> invalid_edge_filter_source_;
  std::vector<int> invalid_edge_filter_target_;

  float sparse_cov_regularization_factor;
};
}  // namespace fast_gicp

#endif
