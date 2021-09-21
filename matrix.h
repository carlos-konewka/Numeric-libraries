//LICENCE
/*
 Copyright 2021 Paweł Gozdur<pawel.gozdur@student.uj.edu.pl>
 or <pawel1216@interia.pl>
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/* Matrix with some bacis operations: det, transpose etc. */

#pragma once
#include <cstdlib>
#include <fstream>

//Headers used in definitions
#include <cstring>
#include <stack>
#include <stdexcept>
#include <algorithm>
#include <string>

template <typename K>
class Matrix {
	public:
	//Constructor creates square matrix filled up with zero.
	Matrix(int n);
	//Constructor creates matrix N x M filled up with zero.
	Matrix(size_t n, size_t m);
	//Constructor, which copies array to matrix (heap matrix).
	Matrix(size_t n, size_t m, K** data);
	//Constructor, which creates matrix from one-dimensional container.
	//Container must have begin() and size() methods. Constructor
	//throw std::range_error exception if container size != n*m.
	template <typename V>
	Matrix(size_t n, size_t m, const V& v);
	//Constructor, which creates matrix from two-dimensional container.
	//Container must have begin(), end() methods. Constructor
	//throw std::range_error exception if cointener has incorrect
	//dimensions.
	template <typename V>
	Matrix(const V& v);
	//Constructor creates matrix and fills it, with given value.
	Matrix(size_t n, size_t m, K value);
	Matrix(const Matrix& M);
	Matrix& operator=(const Matrix& M);
	Matrix(Matrix&& M);
	Matrix& operator=(Matrix&& M);
	//Creates matrix NxN with diagonal filled up with value.
	static Matrix<K> diagonal(size_t n, K val);
	//Returns the number of rows in matrix.
	size_t rows() const;
	//Returns the number of rows in matrix.
	size_t cols() const;
	//Returns pointer to given row (numbered from 0). Throws
	//std::range_error exception if 'i' is bigger than rows-1.
	//Obviously [][] may behaves unpredictably when K isn't primitive
	//type and depends on K operator[](K* k, unsigned int j).
	K* operator[](unsigned int i) const;
	//Returns pointer to array containing scalars.
	K** get_ptr() const;
	//Returns pointer to given row (rows are numbered from 0).
	K* get_row(size_t no) const;
	//Returns pointer to the copy of given row 
	//(rows are numbered from 0).
	K* copy_row(size_t no, K* ptr) const;
	//Returns pointer to the copy of given column.
	//(columns are numbered from 0).
	K* copy_col(size_t no, K* ptr) const;
	//Matrices addition. Returns reference to updated matrix.
	//Throws std::logic_error exception if dimensions of arrays
	//are different.
	Matrix& operator+=(const Matrix& B);
	//Matrices subtraction. Returns reference to updated matrix.
	//Throws std::logic_error exception if dimensions of arrays
	//are different.
	Matrix& operator-=(const Matrix& B);
	//Matrices multiplication. Returns reference to updated matrix.
	////Throws std::logic_error exception if dimensions of arrays
	//are incorrect (this.cols() != B.rows())
	Matrix& operator*=(const Matrix& B);
	//Multiplication by scalar. Returns reference to updated matrix.
	Matrix& operator*=(K scalar);
	//Unary minus. Copies matrix and converts each value to its 
	//opposite element (in current body).
	Matrix operator-() const;
	//Creates new matrix without row and column, where there is element
	//with index i,j (rows and columns are numbered from 0).
	Matrix cut(int row, int col) const;
	//Returns determinant of matrix. Throws std::logic_error
	//exception if matrix isn't square.
	K det() const;
	//Returns trace (sum of diagonal scalars). Throws std::logic_error
	//exception if matrix isn't square.
	K trace() const;
	//Creates new transposed matrix.
	Matrix transpose() const;
	//Creates matrix of algebraic complements. Throws std::logic_error
	//exception if matrix isn't square.
	Matrix algebraic_complement() const;
	//Returns reversed matrix. We should pass as argument det != 0,
	//otherwise throws std::logic_error. Throws std::logic_error as
	//well if matrix isn't square.
	Matrix<K> reverse(K Det) const;
	//Returns reversed matrix. Throws std::logic_error if matrix isn't
	//square or det == 0.
	Matrix<K> reverse() const;//throw if det == 0
	//Matrices addition. Throws std::logic_error exception if 
	//dimensions of arrays are different.
	template <typename T>
	friend Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B);
	//Matrices multiplication. Throws std::logic_error exception if
	//dimensions of arrays are incorrect (A.cols() != B.rows())
	template <typename T>
	friend Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B);
	//Multiplication by scalar.
	template <typename T>
	friend Matrix<T> operator*(const Matrix<T>& M, T scalar);
	//Print matrix. Values are separated by space and rows
	//are separated by std::endl. At the end, there is std::endl.
	template <typename T>
	friend std::ostream& operator<<(std::ostream& stream,
	const Matrix<T>& M);
	//Destructor frees up memory allocated on the heap.
	~Matrix();
	private:
	//rows
	size_t n;
	//columns
	size_t m;
	//scalars in matrix
	K** data;
	//private constructor, does not copy array
	Matrix(size_t n, size_t m, K** data, int);
};
template <typename K>
Matrix<K> operator-(const Matrix<K>& A, const Matrix<K>& B);
template <typename K>
Matrix<K> operator*(K scalar, const Matrix<K>& M);

//Definitions
template <typename K>
Matrix<K>::Matrix(const int n) :  n(n), m(n) {
	data = new K*[n];
	K zero{};
	for(int i=0; i<n; i++) {
		data[i] = new K[n];
		memset(data[i], zero, n*sizeof(K)); 
	}
}
template <typename K>
Matrix<K>::Matrix(size_t n, size_t m) : n(n), m(m) {
	data = new K*[n];
	K zero{};
	for(int i=0; i<n; i++) {
		data[i] = new K[m];
		memset(data[i], zero, m*sizeof(K)); 
	}
}

template <typename K>
Matrix<K>::Matrix(size_t n, size_t m, K** data) : n(n), m(m) {
	this->data = new K*[n];
	for(int i=0; i<n; i++){
		this->data[i] = new K[m];
		memcpy(this->data[i], data[i], m*sizeof(K));
	}
}
template <typename K>
Matrix<K>::Matrix(size_t n, size_t m, K** data, int)
: n(n), m(m), data(data) {}
template <typename K>
template <typename V>
Matrix<K>::Matrix(size_t n, size_t m, const V& v) {
	if(n*m != v.size()) {
		std::string message = std::string(__func__) +
		" - Container has incorrect size to built matrix";
		throw std::range_error(message.c_str());
	}
	this->n = n;
	this->m = m;
	data = new K*[n];
	auto it = v.begin();
	data = new K*[n];
	for(int i=0; i<n; i++) {
		data[i] = new K[m];
		for(int j=0; j<m; j++) {
			data[i][j] = *it;
			it++;
		}
	}
}

template<typename K>
template <typename V>
Matrix<K>::Matrix(const V& v) {
	std::string message = std::string(__func__) +
	" - Container has incorrect size to built matrix";
	n = v.size();
	if(n == 0) throw std::range_error(message.c_str());
	m = v.begin()->size();
	for(auto it = v.begin(); it != v.end(); it++)
		if(it->size() != m) throw std::range_error(message.c_str());
	data = new K*[n];
	int rowIt = -1;
	for(auto row : v){
		data[++rowIt] = new K[m];
		int valIt = -1;
		for(auto val : row) data[rowIt][++valIt] = val;
	}
}

template<typename K>
Matrix<K>::Matrix(size_t n, size_t m, K value) : n(n), m(m) {
	data = new K*[n];
	for(int i=0; i<n; i++){
		data[i] = new K[m];
		memset(data[i], value, m*sizeof(value));
	}
}
template<typename K>
Matrix<K>::Matrix(const Matrix<K>& M) : n(M.n), m(M.m) {
		data = new K*[n];
		for(int i=0; i<n; i++) {
			data[i] = new K[m];
			memcpy(data[i], M.data[i], m*sizeof(K));
		}
}
template<typename K>
Matrix<K>& Matrix<K>::operator=(const Matrix<K>& M){
	if(this == &M) return *this;
	for(int i=0; i<n; i++) delete[] data[i];
	delete[] data;
	n = M.n;
	m = M.m;
	data = new K*[n];
	for(int i=0; i<n; i++) {
		data[i] = new K[m];
		memcpy(data[i], M.data[i], m*sizeof(K));
	}
	return *this;
}
template<typename K>
Matrix<K> Matrix<K>::diagonal(size_t n, K val){
	K** A = new K*[n];
	K zero{};
	int diag = 0;
	for(int i=0; i<n; i++){
		A[i] = new K[n];
		memset(A[i], zero, n*sizeof(K));
		A[diag][diag] = val;
		diag++;
	}
	return Matrix<K>(n, n, A, 0);
}
template<typename K>
Matrix<K>::Matrix(Matrix&& M) : n(M.n), m(M.m), data(M.data) {
	M.n = M.m = 1;
	M.data = new K*[1];
	M.data[0] = new K[1];
}
template<typename K>
Matrix<K>& Matrix<K>::operator=(Matrix<K>&& M) {
	std::swap(data, M.data);
	std::swap(n, M.n);
	std::swap(m, M.m);
	return *this;
}
template<typename K>
K** Matrix<K>::get_ptr() const { return data; }
template<typename K>
size_t Matrix<K>::rows() const { return n; }
template<typename K>
size_t Matrix<K>::cols() const { return m; }
template<typename K>
K* Matrix<K>::operator[](unsigned int i) const {
	if(i >= n) throw std::range_error(__func__);
	return data[i];
}
template<typename K>
K* Matrix<K>::get_row(size_t no) const { return data[no]; }
template<typename K>	
K* Matrix<K>::copy_row(size_t no, K* ptr) const{
	return memcpy(ptr, data[no], sizeof(K)*m);
}
template<typename K>
K* Matrix<K>::copy_col(size_t no, K* ptr) const {
	for(int i=0; i<n; i++) ptr[i] = data[i][no];
	return ptr;
}
template <typename K>
Matrix<K>& Matrix<K>::operator+=(const Matrix<K>& B){
	if(n != B.rows() || m != B.cols()) {
		std::string message = std::string(__func__) + 
		" - Matrices haven't appropriate dimensions";
		throw  std::logic_error(message.c_str());
	}
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++)	data[i][j] += B.data[i][j];
	}
	return *this;
}
template <typename K>
Matrix<K>& Matrix<K>::operator-=(const Matrix<K>& B){
	if(n != B.rows() || m != B.cols()) {
		std::string message = std::string(__func__) + 
		" - Matrices haven't appropriate dimensions";
		throw  std::logic_error(message.c_str());
	}
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++)	data[i][j] -= B.data[i][j];
	}
	return *this;
}
template <typename K>
Matrix<K>& Matrix<K>::operator*=(const Matrix<K>& B){
	if(m != B.rows()) {
		std::string message = std::string(__func__) + 
		" - Matrices haven't appropriate dimensions";
		throw  std::logic_error(message.c_str());
	}
	Matrix<K> result = (*this)*B;
	std::swap(n, result.n);
	std::swap(m, result.m);
	std::swap(data, result.data);
	return *this; 
}
template <typename K>
Matrix<K>& Matrix<K>::operator*=(K scalar){
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++)	data[i][j] *= scalar;
	}
	return *this;
}
template <typename K>
Matrix<K> Matrix<K>::operator-() const {
	K** A = new K*[n];
	for(int i=0; i<n; i++){
		A[i] = new K[m];
		for(int j=0; j<m; j++)	A[i][j] = -data[i][j];
	}
	return Matrix(n, m, A, 0);
}
template <typename K>
Matrix<K> Matrix<K>::cut(int row, int col) const {
	K** A = new K*[n-1];
	int rowIt = -1;
	for(int i=0; i<n; i++){
		if(i == row) continue;
		A[++rowIt] = new K[m-1];
		int colIt = -1;
		for(int j=0; j<m; j++){
			if(j == col) continue;
			A[rowIt][++colIt] = data[i][j];
		}
	}
	return Matrix<K>(n-1, m-1, A, 0);
}
template <typename K>
K Matrix<K>::det() const {
	std::string message = std::string(__func__) + 
	" - Matrix isn't square";
	if(n != m) throw  std::logic_error(message.c_str());
	if(n == 1) return data[0][0];
	if(n == 2) return data[0][0]*data[1][1]-data[0][1]*data[1][0];
	K sum{};
	std::stack<Matrix<K>> M;
	std::stack<K> S;
	for(int i=0; i<m; i++) {
		S.push((i%2) ? -data[0][i] : data[0][i]);
		M.push(cut(0, i));
	}
	while(!M.empty()){
		Matrix<K> currM = M.top();
		M.pop();
		K currS = S.top();
		S.pop();
		if(currM.n == 2) {
			sum += currS*(currM.data[0][0]*currM.data[1][1]-
			currM.data[0][1]*currM.data[1][0]);
		} else {
			K* row = get_row(0);
			for(int i=0; i<currM.m; i++){
				S.push(currS*row[i]);
				M.push(currM.cut(0, i));
			}
		}
	}
	return sum;
}
template <typename K>
K Matrix<K>::trace() const {
	std::string message = std::string(__func__) + 
	" - trace() - Matrix isn't square";
	if(n != m) throw  std::logic_error(message.c_str());
	K sum{};
	for(int i=0; i<n; i++) sum += data[i][i];
	return sum;
}
template <typename K>
Matrix<K> Matrix<K>::transpose() const {
	K** A = new K*[m];
	for(int i=0; i<m; i++) A[i] = new K[n];
	for(int i=0; i<m; i++){
		A[i] = new K[m];
		for(int j=0; j<n; j++) A[i][j] = data[j][i];
	}
	return Matrix(m, n, A, 0);
}
template <typename K>
Matrix<K> Matrix<K>::algebraic_complement() const {
	std::string message = __func__ +
	" - Matrix isn't square";
	if(n != m) throw  std::logic_error(message.c_str());
	K** A = new K*[n];
	for(int i=0; i<n; i++){
		A[i] = new K[m];
		for(int j=0; j<m; j++){
			K d = cut(i, j).det();
			A[i][j] = ((i+j)%2) ? -d : d;
		}
	}
	return Matrix(n, m, A, 0);
}
template <typename K>
Matrix<K> Matrix<K>::reverse(K Det) const {
	std::string message1 = std::string(__func__) +
	" - Matrix isn't square";
	std::string message2 = std::string(__func__) +
	" - Det == 0";
	if(n != m) throw  std::logic_error(message1.c_str());
	if(Det == K{}) throw  std::logic_error(message2.c_str());
	Matrix<K> complement = algebraic_complement();
	Matrix<K> trans = complement.transpose();
	return (Det/Det/Det)*trans;
}
template <typename K>
Matrix<K> Matrix<K>::reverse() const {
	K d = det();
	return reverse(d);
}
//out-of class functions
template <typename K>
Matrix<K> operator*(const Matrix<K>& A, const Matrix<K>& B){
	int nA = A.rows(), mA = A.cols();
	int nB = B.rows(), mB = B.cols();
	K** A_data = A.get_ptr(), **B_data = B.get_ptr();
	if(mA != nB) {
		std::string message = std::string(__func__) +
		" - Matrices haven't appropriate dimensions";
		throw  std::logic_error(message.c_str());
	}
	K** result = new K*[nA];
	for(int i=0; i<nA; i++){
		result[i] = new K[mB];
		for(int j=0; j<mB; j++){
			K sum{};//sum j-ta kolumna z B i i-ty wiersz z A
			for(int t=0; t<mA; t++) sum += A_data[i][t]*B_data[t][j];
			result[i][j] = sum;
		}
	}
	return Matrix<K>(nA, mB, result, 0);
}
template <typename K>
Matrix<K> operator+(const Matrix<K>& A, const Matrix<K>& B){
	int na = A.rows(), ma = A.cols();
	if(na != B.rows() || ma != B.cols()){
		std::string message = std::string(__func__) +
		" - Matrices haven't appropriate dimensions";
		throw  std::logic_error(message.c_str());
	}
	K **A_data = A.get_ptr(), **B_data = B.get_ptr();
	K** result = new K*[na];
	for(int i=0; i<na; i++){
		result[i] = new K[ma];
		for(int j=0; j<ma; j++) 
			result[i][j] = A_data[i][j] + B_data[i][j];
	}
	return Matrix<K>(na, ma, result, 0);
}
template <typename K>
Matrix<K> operator-(const Matrix<K>& A, const Matrix<K>& B){
	return A + (-B);
}
template <typename K>
Matrix<K> operator*(K scalar, const Matrix<K>& M){
	return M*scalar;
}
template <typename K>
Matrix<K> operator*(const Matrix<K>& M, K scalar){
	int n = M.rows(), m = M.cols();
	K** M_data = M.get_ptr();
	K** A = new K*[n];
	for(int i=0; i<n; i++){
		A[i] = new K[m];
		for(int j=0; j<m; j++) A[i][j] = scalar*M_data[i][j];
	}
	return Matrix<K>(n, m, A, 0);
}
template <typename K>
Matrix<K>::~Matrix(){
	for(int i=0; i<n; i++) delete[] data[i];
	delete[] data;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, 
const Matrix<T>& M){
	for(int i=0; i<M.n; i++){
		stream<<M.data[i][0];
		for(int j=1; j<M.m; j++) stream<<" "<<M.data[i][j];
		stream<<std::endl;	
	}
	return stream;
}
