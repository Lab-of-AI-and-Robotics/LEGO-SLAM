#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>

namespace fast_gicp {

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 10;  //25
  reg_name_ = "FastGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();
  knn_max_distance_ = 99999.0;
  
  source_covs_.clear();  
  source_rotationsq_.clear();
  source_scales_.clear();
  source_z_values_.clear();

  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();

  regularization_method_ = RegularizationMethod::NORMALIZED_ELLIPSE;
  search_source_.reset(new SearchMethodSource);
  search_source_sparse_.reset(new SearchMethodSource);
  search_target_.reset(new SearchMethodTarget);
  search_target_edge_.reset(new SearchMethodTarget);

  matching_weight_edge_ = 1.0;
  matching_weight_sparse_ = 1.0;
  source_cov_mid_ = false;

  edge_corr_dist_threshold_ = 0.12; // 0.12
  sparse_cov_regularization_factor = 10.;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::~FastGICP() {}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setNumThreads(int n) {
  num_threads_ = n;


#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setKNNMaxDistance(float k) {
  knn_max_distance_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setEdgeWeight(float weight) {
  matching_weight_edge_ = weight;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSparseWeight(float weight) {
  matching_weight_sparse_ = weight;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovMid(bool use_mid) {
  source_cov_mid_ = use_mid;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);
  source_rotationsq_.swap(target_rotationsq_);
  source_scales_.swap(target_scales_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
  source_rotationsq_.clear();
  source_scales_.clear();

  input_edge_.reset();
  source_edge_covs_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  search_source_->setInputCloud(cloud);
  source_covs_.clear();
  source_rotationsq_.clear();
  source_scales_.clear();

  // for safe implementation
  // check sparse/edge size before matching!!
  input_sparse_.reset();
  input_edge_.reset();
}


template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSourceSparse(const PointCloudSourceConstPtr& cloud) {

  input_sparse_ = cloud;
  search_source_sparse_->setInputCloud(input_sparse_);
}


template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateSourceCovariance() {
	if (input_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	source_covs_.clear();
	source_rotationsq_.clear();
	source_scales_.clear();
	calculate_covariances(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_, source_knn_idxs_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  search_target_->setInputCloud(cloud);
  target_covs_.clear();
  target_rotationsq_.clear();
  target_scales_.clear();
  // std::cout<<"set input target end"<<std::endl;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovariance() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_knn_idxs_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithZ() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
	calculate_covariances_withz(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_z_values_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculateTargetCovarianceWithFilter() {
	if (target_->size() == 0){
		std::cerr<<"no point cloud"<<std::endl;
		return;
	}
	target_covs_.clear();
  target_edge_covs_.clear();
	target_rotationsq_.clear();
	target_scales_.clear();
  target_edge_.reset();
  is_edge_points_target_.clear();
	calculate_target_covariances_with_filter(target_, target_edge_, *search_target_, *search_target_edge_, target_covs_, target_edge_covs_, target_rotationsq_, target_scales_, target_filter_, is_edge_points_target_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales)
	{
		setCovariances(input_rotationsq, input_scales, source_covs_, source_rotationsq_, source_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceZvalues(const std::vector<float>& input_z_values)
	{
		source_z_values_.clear();
    source_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    source_num_trackable_points_ = num_trackable_points;
		source_filter_.clear();
    source_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetFilter(const int num_trackable_points, const std::vector<int>& input_filter)
	{
    target_num_trackable_points_ = num_trackable_points;
		target_filter_.clear();
    target_filter_ = input_filter;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetZvalues(const std::vector<float>& input_z_values)
	{
		target_z_values_.clear();
    target_z_values_ = input_z_values;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales)
	{
		setCovariances(input_rotationsq, input_scales, target_covs_, target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariancesAdditional(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
  const std::vector<int>& input_edge_mask)
	{
		setCovariancesAdditional<PointTarget>(input_rotationsq, input_scales, target_covs_, 
                              input_edge_mask, target_edge_, target_edge_covs_,
                              target_rotationsq_, target_scales_);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {
    throw std::invalid_argument("FastGICP: destination cloud cannot be identical to source or target");
  }
  if (source_covs_.size() != input_->size()) {
    // std::cout<<"compute source cov"<<std::endl;
    // calculate_covariances(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_);
    // calculate_covariances_withz(input_, *search_source_, source_covs_, source_rotationsq_, source_scales_, source_z_values_);
    calculate_source_covariances_with_filter(input_, input_edge_, *search_source_, source_covs_, source_edge_covs_, source_rotationsq_, source_scales_, source_filter_, is_edge_points_source_);
  }
  if (target_covs_.size() != target_->size()) {
    // std::cout<<"compute target cov"<<std::endl;
    calculate_covariances(target_, *search_target_, target_covs_, target_rotationsq_, target_scales_, target_knn_idxs_);
  }
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  bool on_edge = false;
  bool on_sparse = false;

  int extended_edge_size = 0;
  int extended_sparse_size = 0;

  if (input_edge_ && !input_edge_->empty() && matching_weight_edge_ > 0. && input_edge_->size() > 5
      && target_edge_ && !target_edge_->empty() && target_edge_->size() > 10) {
    on_edge = true;
    extended_edge_size = input_edge_->size();
  }

  if (input_sparse_ && !input_sparse_->empty()) {
    on_sparse = true;
    extended_sparse_size = input_sparse_->size();
  }

  int base_matches_size = input_->size();
  int size_matches = base_matches_size;


  if (on_edge) {
    size_matches += extended_edge_size;
  }
  if (on_sparse) {
    size_matches += extended_sparse_size;
  }

  correspondences_.resize(base_matches_size);
  correspondences_extended_edge_.resize(extended_edge_size);
  correspondences_extended_sparse_.resize(extended_sparse_size);
  sq_distances_.resize(base_matches_size);
  mahalanobis_.resize(size_matches);

  // std::cout << "[update correspondence]" << std::endl;
  // std::cout << base_matches_size << "/" << extended_edge_size << "/" << extended_sparse_size << std::endl;

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

  // base/edge/sparse/prev

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < size_matches; i++) {

    // base gicp
    if (i < base_matches_size) {
      PointTarget pt;
    
      pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();
      
      // if (!pcl::isFinite(pt)){
      //   // std::cout << trans_f.data() << std::endl;
      //   // std::cout << pt.x << pt.y << pt.z << std::endl;
      //   continue;
      // }
      
      search_target_->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      
      sq_distances_[i] = k_sq_dists[0];
      correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;
      if (correspondences_[i] < 0) {
        continue;
      }

      const int target_index = correspondences_[i];
      const auto& cov_A = source_covs_[i];
      const auto& cov_B = target_covs_[target_index];

      Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
      RCR(3, 3) = 1.0;

      if (RCR.determinant() == 0){
        // std::cout << "mahalanobis value will be NaN" << std::endl;
        mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
      else{
        mahalanobis_[i] = RCR.inverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
    } 
    else if (i >= base_matches_size && i < base_matches_size+extended_edge_size) { // edge gicp
      int edge_idx = i - base_matches_size;
      PointTarget pt;
    
      pt.getVector4fMap() = trans_f * input_edge_->at(edge_idx).getVector4fMap();
      
      search_target_edge_->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      
      // sq_distances_[i] = k_sq_dists[0];
      correspondences_extended_edge_[edge_idx] = k_sq_dists[0] < edge_corr_dist_threshold_ * edge_corr_dist_threshold_ ? k_indices[0] : -1;
      if (correspondences_extended_edge_[edge_idx] < 0) {
        continue;
      }

      const int target_index = correspondences_extended_edge_[edge_idx];
      const auto& cov_A = source_edge_covs_[edge_idx];
      const auto& cov_B = target_edge_covs_[target_index];

      Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
      RCR(3, 3) = 1.0;

      if (RCR.determinant() == 0){
        // std::cout << "mahalanobis value will be NaN" << std::endl;
        mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
      else{
        mahalanobis_[i] = RCR.inverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
    }
    else if (i >= base_matches_size+extended_edge_size && i < base_matches_size+extended_edge_size+extended_sparse_size) { // sparse gicp
      int sparse_idx = i - base_matches_size - extended_edge_size;
      PointTarget pt;
    
      pt.getVector4fMap() = trans_f * input_sparse_->at(sparse_idx).getVector4fMap();
      
      search_target_->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      
      // sq_distances_[i] = k_sq_dists[0];
      correspondences_extended_sparse_[sparse_idx] = k_sq_dists[0] < edge_corr_dist_threshold_ * edge_corr_dist_threshold_ ? k_indices[0] : -1;
      if (correspondences_extended_sparse_[sparse_idx] < 0) {
        continue;
      }

      const int target_index = correspondences_extended_sparse_[sparse_idx];
      const auto& cov_A = source_sparse_covs_[sparse_idx];
      const auto& cov_B = target_covs_[target_index];

      Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
      RCR(3, 3) = 1.0;

      if (RCR.determinant() == 0){
        // std::cout << "mahalanobis value will be NaN" << std::endl;
        mahalanobis_[i] = RCR.completeOrthogonalDecomposition().pseudoInverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
      else{
        mahalanobis_[i] = RCR.inverse();
        mahalanobis_[i](3, 3) = 0.0f;
      }
    }
  }
  // std::cout << "[update correspondence] done!" << std::endl;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

  bool on_edge = false;
  bool on_sparse = false;

  int extended_edge_size = 0;
  int extended_sparse_size = 0;

  if (input_edge_ && !input_edge_->empty() && matching_weight_edge_ > 0. && input_edge_->size() > 5
      && target_edge_ && !target_edge_->empty() && target_edge_->size() > 10) {
    on_edge = true;
    extended_edge_size = input_edge_->size();
  }

  if (input_sparse_ && !input_sparse_->empty() && matching_weight_sparse_ > 0.) {
    on_sparse = true;
    extended_sparse_size = input_sparse_->size();
  }

  int base_matches_size = input_->size();
  int size_matches = base_matches_size;


  if (on_edge) {
    size_matches += extended_edge_size;
  }
  if (on_sparse) {
    size_matches += extended_sparse_size;
  }

  // std::cout << size_matches << "/" << base_matches_size << "/" << extended_edge_size << "/" << extended_sparse_size << std::endl;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < size_matches; i++) {
    if (i < base_matches_size) {
      int target_index = correspondences_[i];
      if (target_index < 0) {
        continue;
      }

      const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
      const auto& cov_A = source_covs_[i];

      const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
      const auto& cov_B = target_covs_[target_index];

      const Eigen::Vector4d transed_mean_A = trans * mean_A;
      const Eigen::Vector4d error = mean_B - transed_mean_A;

      sum_errors += error.transpose() * mahalanobis_[i] * error;

      if (H == nullptr || b == nullptr) {
        continue;
      }

      Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
      dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
      dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

      Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
      Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

      Hs[omp_get_thread_num()] += Hi;
      bs[omp_get_thread_num()] += bi;
    } 
    else if (i >= base_matches_size && i < base_matches_size+extended_edge_size) { // edge gicp
      int edge_idx = i - base_matches_size;

      int target_index = correspondences_extended_edge_[edge_idx];
      if (target_index < 0) {
        continue;
      }

      
      const Eigen::Vector4d mean_A = input_edge_->at(edge_idx).getVector4fMap().template cast<double>();
      const auto& cov_A = source_edge_covs_[edge_idx];

      const Eigen::Vector4d mean_B = target_edge_->at(target_index).getVector4fMap().template cast<double>();
      const auto& cov_B = target_edge_covs_[target_index];

      const Eigen::Vector4d transed_mean_A = trans * mean_A;
      const Eigen::Vector4d error = mean_B - transed_mean_A;

      sum_errors += matching_weight_edge_ * error.transpose() * mahalanobis_[i] * error;

      if (H == nullptr || b == nullptr) {
        continue;
      }

      Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
      dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
      dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

      Eigen::Matrix<double, 6, 6> Hi = matching_weight_edge_ * jlossexp.transpose() * mahalanobis_[i] * jlossexp;
      Eigen::Matrix<double, 6, 1> bi = matching_weight_edge_ * jlossexp.transpose() * mahalanobis_[i] * error;

      Hs[omp_get_thread_num()] += Hi;
      bs[omp_get_thread_num()] += bi;
    }
    else if (i >= base_matches_size+extended_edge_size && i < base_matches_size+extended_edge_size+extended_sparse_size) { // sparse gicp
      int sparse_idx = i - base_matches_size - extended_edge_size;

      int target_index = correspondences_extended_sparse_[sparse_idx];
      if (target_index < 0) {
        continue;
      }

      
      const Eigen::Vector4d mean_A = input_sparse_->at(sparse_idx).getVector4fMap().template cast<double>();
      const auto& cov_A = source_sparse_covs_[sparse_idx];

      const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
      const auto& cov_B = target_covs_[target_index];

      const Eigen::Vector4d transed_mean_A = trans * mean_A;
      const Eigen::Vector4d error = mean_B - transed_mean_A;

      sum_errors += matching_weight_sparse_ * error.transpose() * mahalanobis_[i] * error;

      if (H == nullptr || b == nullptr) {
        continue;
      }

      Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
      dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
      dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

      Eigen::Matrix<double, 6, 6> Hi = matching_weight_sparse_ * jlossexp.transpose() * mahalanobis_[i] * jlossexp;
      Eigen::Matrix<double, 6, 1> bi = matching_weight_sparse_ * jlossexp.transpose() * mahalanobis_[i] * error;

      Hs[omp_get_thread_num()] += Hi;
      bs[omp_get_thread_num()] += bi;
    
    }

    
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& knn_idxs
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());
  knn_idxs.resize(10*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
        knn_idxs[10*i + j] = k_indices[j];
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances_withz(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<float>& z_values
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    // int num_reliable_neighbors = k_correspondences_;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    float z = std::max(1.,pow(z_values[i], 1.5)*2.);
    // std::cout<<z<<std::endl;
    scales[3*i+0] = (float)scale.x()/z;
    scales[3*i+1] = (float)scale.y()/z;
    scales[3*i+2] = (float)scale.z()/z;

    // compute regularized covariance
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_ELLIPSE:
          // std::cout<<svd.singularValues()(1)<<std::endl;
          if (svd.singularValues()(1) == 0){
          	values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
          }
          else{          
            values = svd.singularValues() / svd.singularValues()(1);
            values = values.array().max(1e-3);
	        }
          break;

      }
      // use regularized covariance
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }
  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_source_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  PointCloudSourcePtr& edge_cloud,
  pcl::search::Search<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter,
  std::vector<int>& is_edge_points
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  
  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(source_num_trackable_points_);
  // pcl::copyPointCloud(*cloud, filter, *newCloud);

  // save covariances of trackable points
  covariances.resize(source_num_trackable_points_);
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());
  source_knn_idxs_.resize(10*cloud->size());
  is_edge_points.resize(cloud->size());

  std::vector<std::vector<PointT>> local_edge_points(num_threads_);
  std::vector<std::vector<Eigen::Matrix4d>> local_edge_covs(num_threads_);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
    
    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
        source_knn_idxs_[10*i + j] = k_indices[j];
      }
    }

    
    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariance and xyz of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1e-3, 1e-3, 1e-5); //1,1,1e-3
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            // std::cout<<svd.singularValues()(1)<<std::endl;
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              if (source_cov_mid_) values = svd.singularValues() / svd.singularValues()(1);
              else values = svd.singularValues() / svd.singularValues().maxCoeff();
              // values = values.array().max(1e-3);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        }
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);


        const float lambda1 = (float)scale.x();
        const float lambda2 = (float)scale.y();
        const float lambda3 = (float)scale.z();

        const float linearity = (lambda1 - lambda2) / lambda1;
        const float planarity = (lambda2 - lambda3) / lambda1;
        const float curvature = lambda3 / (lambda1 + lambda2 + lambda3);

        // if (linearity > 0.4 && planarity < 0.4 && curvature > 0.05 && !is_valid_point[i]) {
        // std::cout << invalid_edge_filter_source_[i] << std::endl;
        if (invalid_edge_filter_source_[i] != 1 && linearity > 0.4 && planarity < 0.4 && curvature > 0.05) {
          const int thread_id = omp_get_thread_num();
          local_edge_points[thread_id].push_back(cloud->at(i));
          local_edge_covs[thread_id].push_back(covariances[filter[i]-1]);
          is_edge_points[i] = 1;
        } else {
          is_edge_points[i] = 0;
        }

      }
    }
  }
  size_t total_edge_points = 0;
  for (const auto& vec : local_edge_points) {
    total_edge_points += vec.size();
  }

  edge_cloud.reset(new pcl::PointCloud<PointSource>()); 
  edge_cloud->points.resize(total_edge_points);
  edge_covariances.clear();
  edge_covariances.resize(total_edge_points);

  size_t current_pos = 0;
  for (int i = 0; i < num_threads_; ++i) {
    if (!local_edge_points[i].empty()) {
      
      std::copy(local_edge_points[i].begin(), local_edge_points[i].end(), edge_cloud->points.begin() + current_pos);
      std::copy(local_edge_covs[i].begin(), local_edge_covs[i].end(), edge_covariances.begin() + current_pos);
      current_pos += local_edge_points[i].size();
    }
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(newCloud);
  
  search_source_->setInputCloud(newCloud);

  // sparse points
  if (input_sparse_ && !input_sparse_->empty()) {
    source_sparse_covs_.resize(input_sparse_->size());

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for (int i = 0; i < input_sparse_->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      int num_reliable_neighbors = 0;
      search_source_sparse_->nearestKSearch(input_sparse_->at(i), k_correspondences_, k_indices, k_sq_distances);
      
      // Get number of reliable neighbors
      for (int j = 0; j < k_indices.size(); j++) {
        if (k_sq_distances[j] < knn_max_distance_){
          ++num_reliable_neighbors;
        }
      }

      Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
      for (int j = 0; j < k_indices.size(); j++) {
        if (k_sq_distances[j] < knn_max_distance_){
          neighbors.col(j) = input_sparse_->at(k_indices[j]).getVector4fMap().template cast<double>();
          source_knn_idxs_[10*i + j] = k_indices[j];
        }
      }

      
      neighbors.colwise() -= neighbors.rowwise().mean().eval();
      Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
      
      //compute raw scale and quaternions using cov
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);

      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        source_sparse_covs_[i] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        source_sparse_covs_[i].setZero();
        source_sparse_covs_[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1e-3, 1e-3, 1e-5); //1,1,1e-3
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            // std::cout<<svd.singularValues()(1)<<std::endl;
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{
              // values = svd.singularValues() / svd.singularValues()(1);
              values = svd.singularValues() / svd.singularValues().maxCoeff();
              values *= sparse_cov_regularization_factor;
              // values = values.array().max(1e-3);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        }
        // use regularized covariance
        source_sparse_covs_[i].setZero();
        source_sparse_covs_[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
      }
    }
  }

  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_target_covariances_with_filter(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  PointCloudTargetPtr& edge_cloud,
  pcl::search::Search<PointT>& kdtree,
  pcl::search::Search<PointT>& edge_kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales,
  std::vector<int>& filter,
  std::vector<int>& is_edge_points
  ) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }

  typename pcl::PointCloud<PointT>::Ptr newCloud(new pcl::PointCloud<PointT>);
  newCloud->points.resize(target_num_trackable_points_);
  // pcl::copyPointCloud(*cloud, filter, *newCloud);

  // save covariances of trackable points
  covariances.resize(target_num_trackable_points_);
  // calculate and save rot/scales about all points
  rotationsq.resize(4*cloud->size());
  scales.resize(3*cloud->size());
  is_edge_points.resize(cloud->size());

  std::vector<std::vector<PointT>> local_edge_points(num_threads_);
  std::vector<std::vector<Eigen::Matrix4d>> local_edge_covs(num_threads_);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    int num_reliable_neighbors = 0;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    // Get number of reliable neighbors
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        ++num_reliable_neighbors;
      }
    }

    Eigen::Matrix<double, 4, -1> neighbors(4, num_reliable_neighbors);
    for (int j = 0; j < k_indices.size(); j++) {
      if (k_sq_distances[j] < knn_max_distance_){
        neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
      }
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    
    //compute raw scale and quaternions using cov
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Quaterniond qfrommat(svd.matrixU());
    qfrommat.normalize();
//    Eigen::Vector4d q = {qfrommat.x(), qfrommat.y(), qfrommat.z(), qfrommat.w()};
    // rotationsq.insert(rotationsq.end(), { (float)qfrommat.x(), (float)qfrommat.y(), (float)qfrommat.z(), (float)qfrommat.w()});
    rotationsq[4*i+0] = (float)qfrommat.x();
    rotationsq[4*i+1] = (float)qfrommat.y();
    rotationsq[4*i+2] = (float)qfrommat.z();
    rotationsq[4*i+3] = (float)qfrommat.w();
    Eigen::Vector3d scale = svd.singularValues().cwiseSqrt();
    // scales.insert(scales.end(), {(float)scale.x(), (float)scale.y(), (float)scale.z()});
    scales[3*i+0] = (float)scale.x();
    scales[3*i+1] = (float)scale.y();
    scales[3*i+2] = (float)scale.z();

    // Save covariances of trackable points
    if (filter[i]!=0){
      // compute regularized covariance
      if (regularization_method_ == RegularizationMethod::NONE) {
        covariances[filter[i]-1] = cov;
      } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
        double lambda = 1e-3;
        Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d C_inv = C.inverse();
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
      } else {
        // Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d values;
        switch (regularization_method_) {
          default:
            std::cerr << "you need to set method (ex: RegularizationMethod::PLANE)" << std::endl;
            abort();
          case RegularizationMethod::PLANE:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
            break;
          case RegularizationMethod::MIN_EIG:
            values = svd.singularValues().array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_MIN_EIG:
            values = svd.singularValues() / svd.singularValues().maxCoeff();
            values = values.array().max(1e-3);
            break;
          case RegularizationMethod::NORMALIZED_ELLIPSE:
            // std::cout<<svd.singularValues()(1)<<std::endl;
            if (svd.singularValues()(1) == 0){
              values = Eigen::Vector3d(1e-9, 1e-9, 1e-9);
            }
            else{          
              // values = svd.singularValues() / svd.singularValues()(1);
              values = svd.singularValues() / svd.singularValues().maxCoeff();
              // values = values.array().max(1e-3);
            }
            break;
          case RegularizationMethod::TEST:
            values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        }
        // use regularized covariance
        covariances[filter[i]-1].setZero();
        covariances[filter[i]-1].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        newCloud->points[filter[i]-1] = cloud->at(i);

        const float lambda1 = (float)scale.x();
        const float lambda2 = (float)scale.y();
        const float lambda3 = (float)scale.z();

        const float linearity = (lambda1 - lambda2) / lambda1;
        const float planarity = (lambda2 - lambda3) / lambda1;
        const float curvature = lambda3 / (lambda1 + lambda2 + lambda3);

        // if (linearity > 0.4 && planarity < 0.4 && curvature > 0.05 && !is_valid_point[i]) {
        if (!invalid_edge_filter_target_[i] && linearity > 0.4 && planarity < 0.4 && curvature > 0.05) {
          const int thread_id = omp_get_thread_num();
          local_edge_points[thread_id].push_back(cloud->at(i));
          local_edge_covs[thread_id].push_back(covariances[filter[i]-1]);
          is_edge_points[i] = 1;
        } else {
          is_edge_points[i] = 0;
        }
      }
    }
  }
  // end for
  size_t total_edge_points = 0;
  for (const auto& vec : local_edge_points) {
    total_edge_points += vec.size();
  }

  edge_cloud.reset(new pcl::PointCloud<PointSource>()); 
  edge_cloud->points.resize(total_edge_points);
  edge_covariances.clear();
  edge_covariances.resize(total_edge_points);

  size_t current_pos = 0;
  for (int i = 0; i < num_threads_; ++i) {
    if (!local_edge_points[i].empty()) {
      
      std::copy(local_edge_points[i].begin(), local_edge_points[i].end(), edge_cloud->points.begin() + current_pos);
      std::copy(local_edge_covs[i].begin(), local_edge_covs[i].end(), edge_covariances.begin() + current_pos);
      current_pos += local_edge_points[i].size();
    }
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(newCloud);
  search_target_->setInputCloud(newCloud);
  if (total_edge_points > 10) {
    search_target_edge_->setInputCloud(edge_cloud);
  }
  
  // std::cout << "Cloud size : " << newCloud->size() << "/cov size : " << covariances.size() << "/rots size : " << rotationsq.size()/4 << std::endl;
  // std::cout << "Checker : " << checker << std::endl;
  // std::cout << "edge cloud size: " << edge_cloud->points.size() << " cov: " << edge_covariances.size() << std::endl;

  return true;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCovariances(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d,
  Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales) 
	{
	if(input_rotationsq.size()/4 != input_scales.size()/3){
		std::cerr << "size not match" <<std::endl;
		abort();
	}
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq;
	scales = input_scales;
  // covariances.resize(input_scales.size());
	covariances.resize(input_scales.size()/3);
	// rotationsq.resize(input_scales.size());
	// scales.resize(input_scales.size());
  // clock_t start_time = clock();
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<scales.size()/3; i++){
		Eigen::Vector3d singular_values = { (double)scales[3*i+0]*scales[3*i+0], 
							(double)scales[3*i+1]*scales[3*i+1], 
							(double)scales[3*i+2]*scales[3*i+2] };
		switch (regularization_method_) {
		default:
		  std::cerr << "here must not be reached" << std::endl;
		  abort();
		case RegularizationMethod::PLANE:
		  singular_values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
		  break;
		case RegularizationMethod::MIN_EIG:
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_MIN_EIG:
		  singular_values = singular_values / singular_values.maxCoeff();
		  singular_values = singular_values.array().max(1e-3);
		  break;
		case RegularizationMethod::NORMALIZED_ELLIPSE:
		  // std::cout<<svd.singularValues()(1)<<std::endl;
		  if (singular_values(1) < 1e-3){
		  	singular_values = Eigen::Vector3d(1e-3, 1e-3, 1e-3);
		  }
		  else{          
			  // singular_values = singular_values / singular_values(1);
        singular_values = singular_values / singular_values.maxCoeff();
			  // singular_values = singular_values.array().max(1e-3);
		  }
		  break;
    case RegularizationMethod::TEST:
      // singular_values = singular_values / singular_values(1) * 1e-2;
      break;
		case RegularizationMethod::NONE:
		  // do nothing
		  break;
		case RegularizationMethod::FROBENIUS:
		  std::cerr<< "should be implemented"<< std::endl;
		  abort();
	      }
	      // scales[i] = singular_values.cwiseSqrt();
	      // rotationsq[i] = input_rotationsq[i];
	      Eigen::Quaterniond q( (double)rotationsq[4*i+0], 
	      				(double)rotationsq[4*i+1], 
	      				(double)rotationsq[4*i+2], 
	      				(double)rotationsq[4*i+3]);
	      q = q.normalized();
	      covariances[i].setZero();
	      covariances[i].template block<3, 3>(0, 0) = q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
  }
}


template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
void FastGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCovariancesAdditional(
	const std::vector<float>& input_rotationsq,
	const std::vector<float>& input_scales,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  const std::vector<int>& input_edge_mask,
  PointCloudTargetPtr& edge_cloud,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& edge_covariances,
  std::vector<float>& rotationsq,
  std::vector<float>& scales)
	{
	if(input_rotationsq.size()/4 != input_scales.size()/3){
		std::cerr << "size not match" <<std::endl;
		abort();
	}
	rotationsq.clear();
	scales.clear();
	rotationsq = input_rotationsq;
	scales = input_scales;
	covariances.resize(input_scales.size()/3);
  
  std::vector<std::vector<PointT>> local_edge_points(num_threads_);
  std::vector<std::vector<Eigen::Matrix4d>> local_edge_covs(num_threads_);

  // all points
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
	for(int i=0; i<scales.size()/3; i++){
		Eigen::Vector3d singular_values = { (double)scales[3*i+0]*scales[3*i+0], 
							(double)scales[3*i+1]*scales[3*i+1], 
							(double)scales[3*i+2]*scales[3*i+2] };
		switch (regularization_method_) {
      default:
        std::cerr << "here must not be reached" << std::endl;
        abort();
      case RegularizationMethod::PLANE:
        singular_values = Eigen::Vector3d(1e-2, 1e-2, 1e-5); //1,1,1e-3
        break;
      case RegularizationMethod::MIN_EIG:
        singular_values = singular_values.array().max(1e-3);
        break;
      case RegularizationMethod::NORMALIZED_MIN_EIG:
        singular_values = singular_values / singular_values.maxCoeff();
        singular_values = singular_values.array().max(1e-3);
        break;
      case RegularizationMethod::NORMALIZED_ELLIPSE:
        // std::cout<<svd.singularValues()(1)<<std::endl;
        // if (singular_values(1) < 1e-3){
        //   singular_values = Eigen::Vector3d(1e-3, 1e-3, 1e-3);
        // }
        // else{
          // singular_values = singular_values / singular_values(1);
          singular_values = singular_values / singular_values.maxCoeff() * 0.1;
          // singular_values = singular_values / singular_values.maxCoeff();
          // singular_values = singular_values.array().max(1e-3);
        // }
        break;
      case RegularizationMethod::TEST:
        // singular_values = singular_values / singular_values(1) * 1e-2;
        break;
      case RegularizationMethod::NONE:
        // do nothing
        break;
      case RegularizationMethod::FROBENIUS:
        std::cerr<< "should be implemented"<< std::endl;
        abort();
    }

    // scales[i] = singular_values.cwiseSqrt();
    // rotationsq[i] = input_rotationsq[i];
    Eigen::Quaterniond q( (double)rotationsq[4*i+0], 
            (double)rotationsq[4*i+1], 
            (double)rotationsq[4*i+2], 
            (double)rotationsq[4*i+3]);
    q = q.normalized();
    covariances[i].setZero();
    covariances[i].template block<3, 3>(0, 0) = q.toRotationMatrix() * singular_values.asDiagonal() * q.toRotationMatrix().transpose();
  
    if (input_edge_mask[i]) {
      const int thread_id = omp_get_thread_num();
      local_edge_points[thread_id].push_back(target_->at(i));
      local_edge_covs[thread_id].push_back(covariances[i]);
    }
  }

  size_t total_edge_points = 0;
  for (const auto& vec : local_edge_points) {
    total_edge_points += vec.size();
  }

  edge_cloud.reset(new pcl::PointCloud<PointSource>()); 
  edge_cloud->points.resize(total_edge_points);
  edge_covariances.clear();
  edge_covariances.resize(total_edge_points);

  size_t current_pos = 0;
  for (int i = 0; i < num_threads_; ++i) {
    if (!local_edge_points[i].empty()) {
      
      std::copy(local_edge_points[i].begin(), local_edge_points[i].end(), edge_cloud->points.begin() + current_pos);
      std::copy(local_edge_covs[i].begin(), local_edge_covs[i].end(), edge_covariances.begin() + current_pos);
      current_pos += local_edge_points[i].size();
    }
  }

  if (total_edge_points > 10) {
    search_target_edge_->setInputCloud(edge_cloud);
  }
}


}  // namespace fast_gicp

#endif
