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
/*
	Polynomial of one variable above K-body. Body should have defined 
following operators: +, - (unary and binary), +=, -=, *, *=, /. K may be
a primitive type or a class. If body isn't a std::complex, 
integer (short, long etc. ), floating (float, double, long double etc.),
it must have conversion operator from std::complex<long double> to K.
*/

//Libraries used in declarations
#include <cstdlib>
#include <fstream>

//Libraries used in definitions
#include <vector>
#include <cstring>
#include <complex>
#include <stdexcept>
#include <cmath>

template <typename K>
class Polynomial {
	public:
	//Creates P(x) = 0
	Polynomial();
	//Creates polynomial with coefficients a_0 = cof[0], a_1 = cof[1]...
	//a_deg = cof[deg]
	Polynomial(K* cof, size_t deg);
	//Constructor needed for operations: P(x) + val, P(x) * val etc.
	Polynomial(K a0);
	//Container must have iterator with begin(), end() and size() 
	//methods. If v.size == 0, it creates P(x) = 0
	template <typename V>
	Polynomial(const V& v);
	Polynomial(const Polynomial& P);
	Polynomial(Polynomial&& P);
	Polynomial& operator=(const Polynomial& P);
	Polynomial& operator=(Polynomial&& P);
	size_t get_deg() const;
	//P(x) = P(x) + Q(x)
	Polynomial& operator+=(const Polynomial& P);
	//P(x) = P(x) - Q(x)
	Polynomial& operator-=(const Polynomial& P);
	//P(x) = P(x) * Q(x). Simple multiplication for small polynomials
	//and FFT for others O(n*log n).
	Polynomial& operator*=(const Polynomial& P);
	//Unary operator to negate coefficients. -P(x)
	Polynomial operator-() const;
	//Returns reference to a_i coefficient. Throws
	//std::range_error exception if i > deg.
	K& operator[](unsigned int i) const;
	//Returns value for passed argument: P(arg). Runs in O(n) time.
	template <typename T>
	T operator()(T arg) const;
	//Recomended way to divide polynomial and gain rest, beacuse using
	// / and % operators separately increases time twice. Result is 
	//assigned to result, and rest to rest.
	void division(const Polynomial& divider,
	Polynomial& result, Polynomial& rest) const;
	//Destructos frees up memory on heap.
	~Polynomial();
	//P(x) * Q(x)
	template <typename T>
	friend Polynomial<T> operator*(const Polynomial<T>& A, 
		const Polynomial<T>& B);
	//P(x) + Q(x)
	template <typename T>
	friend Polynomial<T> operator+(const Polynomial<T>& A,
		const Polynomial<T>& B);
	//P(x) - Q(x)
	template <typename T>
	friend Polynomial<T> operator-(const Polynomial<T>& A,
		const Polynomial<T>& B);
	//P(x) % Q(x)
	template <typename T>
	friend Polynomial<T> operator%(const Polynomial<T>& A,
		const Polynomial<T>& B);
	//P(x) / Q(x)
	template <typename T>
	friend Polynomial<T> operator/(const Polynomial<T>& A,
		const Polynomial<T>& B);
	//Writes out coeficients a_0, a_1 ... a_deg with single
	//space between them.
	template <typename T>
	friend std::ostream& operator<<(std::ostream& o, 
		const Polynomial<T>& P);
	//Returns P(w_0), P(w_1), ... , P(w_n-1) values, where
	//w_0, w_1,... are successive n-degree complex roots of 
	//the number 1. n is the smallest number such as
	//n > deg and n_min <= n. n will be in the form 2^k
	std::vector<std::complex<long double> > DFT(int n_min = 1) const;
	private:
	size_t deg;
	K* cof;
	//Private constructor, it just assigns cof to this->cof.
	Polynomial(K* cof, size_t deg, int);
	//Returns vector of polynomial values, where args is
	//vector of arguments.
	std::vector<std::complex<long double> > compute(
	const std::vector<std::complex<long double> >& args) const;
	//Works recursively. Method computes P(args[0]), P(args[1]), ...
	//Where deg = 2^k-1, args size() == deg+1 and args_0, args_1,
	//... are successive (deg+1)-degree complex roots of 
	//the number 1.
	static std::vector<std::complex<long double> > DFT(
	const std::vector<std::complex<long double> >& args,
	const Polynomial& P);
	//Multiplication with FFT. Returns polynomial over Complex-body, 
	//which should be transformed to K-body. or T-body.
	template <typename T>
	static std::vector<std::complex<long double> > 
	mull(const Polynomial<K>& A,
	const Polynomial<T>& B);
	//Multiplication runs in O(n^2) time, witch simply mulltiplies each
	//other A_i by B_j. It's iterative and used in 
	//advanced mutliplication methods.
	static Polynomial<K> mullN2(
	const Polynomial<K>& A, const Polynomial<K>& B);
	//Below there are some converting functions for most of primitive
	//data types. They are used in polynomial multiplication (FFT)
	//to convert complex<long double> to K-body scalar. If convert's
	//function (complex -> K) doesn't exist,
	//converion operator (K) should be defined.
	
	template <typename T,
    std::enable_if_t<std::is_integral<T>::value, bool> = true>
	static void convert(const std::complex<long double>& from, 
	T& to);
	
	template <typename T,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
	static void convert(const std::complex<long double>& from, 
	T& to);
	
	static void convert(const std::complex<long double>& from, 
	std::complex<long double>& to);
	
	static void convert(const std::complex<long double>& from, 
	std::complex<double>& to);
	
	static void convert(const std::complex<long double>& from, 
	std::complex<float>& to);
	
	template <typename T,
    std::enable_if_t<!(std::is_integral<T>::value
    || std::is_floating_point<T>::value
    ), bool> = true>
	static void convert(const std::complex<long double>& from, 
	T& to);
};


template <typename K>
Polynomial<K>::Polynomial() : deg(0) {
	cof = new K[1]();
}

template <typename K>
Polynomial<K>::Polynomial(K* cof, size_t deg) : deg(deg) {
	this->cof = new K[deg+1];
	std::memcpy(this->cof, cof, (deg+1)*sizeof(K));
}

template <typename K>
Polynomial<K>::Polynomial(K a0) : deg(0) {
	cof = new K[1];
	cof[0] = a0;	
}

template <typename K>
template <typename V>
Polynomial<K>::Polynomial(const V& v) {
	if(v.size()) {
		deg = v.size() - 1;
		cof = new K[v.size()];
		int cofIt = -1;
		for(auto it = v.begin(); it != v.end(); it++) 
			cof[++cofIt] = *it;
	} else {
		deg = 0;
		cof = new K[1]();
	}
}

template <typename K>
Polynomial<K>::Polynomial(const Polynomial& P) : deg(P.deg) {
	cof = new K[deg+1];
	std::memcpy(cof, P.cof, (deg+1)*sizeof(K));
}

template <typename K>
Polynomial<K>::Polynomial(Polynomial&& P) {
	std::swap(deg, P.deg);
	std::swap(cof, P.cof);
}

template <typename K>
Polynomial<K>::Polynomial(K* cof, size_t deg, int) : 
cof(cof), deg(deg) {
}

template <typename K>
Polynomial<K>& Polynomial<K>::operator=(const Polynomial<K>& P) {
	if(this == &P) return *this;
	delete[] cof;
	deg = P.deg;
	cof = new K[deg+1];
	memcpy(cof, P.cof, (deg+1)*sizeof(K));
	return *this;
}

template <typename K>
Polynomial<K>& Polynomial<K>::operator=(Polynomial<K>&& P) {
	std::swap(deg, P.deg);
	std::swap(cof, P.cof);
	return *this;
}

template <typename K>
size_t Polynomial<K>::get_deg() const {
	return deg;
}

template <typename K>
Polynomial<K>& Polynomial<K>::operator+=(const Polynomial<K>& P) {
	if( deg < P.deg) {
		K* cof2 = new K[P.deg+1];
		for(int i = 0; i <= deg; i++) cof2[i] = cof[i] + P.cof[i];
		std::memcpy(cof2 + deg + 1, 
			P.cof + deg + 1,
			(P.deg - deg)*sizeof(K));
		delete[] cof;
		cof = cof2;
		deg = P.deg;
		return *this;
	} else if(P.deg < deg) {
		for(int i = 0; i <= P.deg; i++)
			cof[i] += P.cof[i];
		return *this;
	}
	int last = deg;
	K zero{};
	while(deg > 0 && cof[deg] == zero) deg--;
	if(last != deg) {
		K* cf = new K[deg + 1];
		std::memcpy(cf, cof, (deg + 1)*sizeof(K));
		delete[] cof;
		cof = cf;
	}
	return *this;
}

template <typename K>
Polynomial<K>& Polynomial<K>::operator-=(const Polynomial<K>& P) {
	if( deg < P.deg) {
		K* cof2 = new K[P.deg+1];
		for(int i = 0; i <= deg; i++) cof2[i] = cof[i] - P.cof[i];
		std::memcpy(cof2 + deg + 1, 
			P.cof + deg + 1,
			(P.deg - deg)*sizeof(K));
		delete[] cof;
		cof = cof2;
		deg = P.deg;
		return *this;
	} else if(P.deg < deg) {
		for(int i = 0; i <= P.deg; i++)
			cof[i] -= P.cof[i];
		return *this;
	}
	int last = deg;
	K zero{};
	while(deg > 0 && cof[deg] == zero) deg--;
	if(last != deg) {
		K* cf = new K[deg + 1];
		std::memcpy(cf, cof, (deg + 1)*sizeof(K));
		delete[] cof;
		cof = cf;
	}
	return *this;
}

template <typename K>
Polynomial<K>& Polynomial<K>::operator*=(const Polynomial<K>& P) {
	Polynomial<K> Q = *this * P;
	std::swap(deg, Q.deg);
	std::swap(cof, Q.cof);
	return *this;
}

template <typename K>
Polynomial<K> Polynomial<K>::operator-() const {
	K* cof2 = new K[deg+1];
	for(int i=0; i <= deg; i++) cof2[i] = -cof[i];
	return Polynomial<K>(cof2, deg+1);
}

template <typename K>
K& Polynomial<K>::operator[](unsigned int i) const {
	if(i > deg) throw std::range_error(__func__);
	return cof[i];
}

template <typename K>
template <typename T>
T Polynomial<K>::operator()(T arg) const {
	T sum = cof[0];
	T x = arg;
	for(int i=1; i <= deg; i++){
		sum += ((T)cof[i])*arg;
		arg *= x;
	}
	return sum;
}


template <typename K>
Polynomial<K>::~Polynomial() {
	delete[] cof;
}

template <typename K>
template <typename T>
std::vector<std::complex<long double> > 
Polynomial<K>::mull(const Polynomial<K>& A,
const Polynomial<T>& B) {
	int deg_max = std::max(A.deg, B.deg);
	int deg_min = std::min(A.deg, B.deg);
	int n_min = 1;
	while(n_min <= (deg_max+deg_min)) n_min *= 2;
	auto a = A.DFT(n_min);
	auto b = B.DFT(n_min);
	if(a.size() != b.size()) {
		std::string message = std::string(__func__) +
		std::string(": A.deg != B.deg");
		throw std::logic_error(message.c_str());
	}
	auto itA = a.begin();
	auto itB = b.begin();
	while(itA != a.end()){
		(*itA) *= (*itB);
		itA++;
		itB++;
	}
	Polynomial<std::complex<long double> > to_reverse(a);
	auto reversed = to_reverse.DFT(n_min);
	long double t = 1./n_min;
	for(auto &rev : reversed) rev *= t;
	int i = 1, j = reversed.size()-1;
	while(i < j) {
		std::swap(reversed[i], reversed[j]);
		i++;
		j--;
	}
	return reversed;
}
template <typename K>
Polynomial<K>
operator*(const Polynomial<K>& A, 
const Polynomial<K>& B) {
	if(A.deg < 300 && B.deg < 300){
		return Polynomial<K>::mullN2(A, B);
	}
	std::vector<std::complex<long double> > v = 
	Polynomial<K>::mull(A, B);
	std::vector<K> res;
	v.erase(v.begin()+A.deg+B.deg+1, v.end());
	for(auto val : v) {
		K tmp;
		Polynomial<K>::convert(val, tmp);
		res.push_back(tmp);
	}
	return Polynomial<K>(res);
}

template <typename K>
Polynomial<K> operator+(const Polynomial<K>& A,
const Polynomial<K>& B) {
	K* cf;
	if(A.deg < B.deg) {
		cf = new K[B.deg + 1];
		for(int i=0; i <= A.deg; i++) cf[i] = A.cof[i] + B.cof[i];
		std::memcpy(cf + A.deg + 1, 
			B.cof + A.deg + 1, 
			(B.deg - A.deg)*sizeof(K));
		return Polynomial<K>(cf, B.deg, 0);
	} else if(A.deg > B.deg) {
		cf = new K[A.deg + 1];
		for(int i=0; i <= B.deg; i++) cf[i] = A.cof[i] + B.cof[i];
		std::memcpy(cf + B.deg + 1, 
			A.cof + B.deg + 1, 
			(A.deg - B.deg)*sizeof(K));
		return Polynomial<K>(cf, A.deg, 0);
	}
	K zero{};
	int Deg = A.deg;//or B.deg
	while(Deg > 0 && A.cof[Deg] + B.cof[Deg] == zero) Deg--;
	cf =  new K[Deg + 1];
	for(int i = 0; i <= Deg; i++) cf[i] = A.cof[i] + B.cof[i];
	return Polynomial<K>(cf, Deg, 0);
}

template <typename K>
Polynomial<K> operator-(const Polynomial<K>& A,
const Polynomial<K>& B) {
	K* cf;
	if(A.deg < B.deg) {
		cf = new K[B.deg + 1];
		for(int i = 0; i <= A.deg; i++) cf[i] = A.cof[i] - B.cof[i];
		for(int i = A.deg + 1; i <= B.deg; i++) cf[i] = -B.cof[i];
		return Polynomial<K>(cf, B.deg, 0);
	} else if(A.deg > B.deg) {
		cf = new K[A.deg + 1];
		for(int i=0; i <= B.deg; i++) cf[i] = A.cof[i] - B.cof[i];
		for(int i = B.deg + 1; i <= A.deg; i++) cf[i] = -A.cof[i];
		return Polynomial<K>(cf, A.deg, 0);
	}
	K zero{};
	int Deg = A.deg;//or B.deg
	while(Deg > 0 && A.cof[Deg] - B.cof[Deg] == zero) Deg--;
	cf =  new K[Deg + 1];
	for(int i = 0; i <= Deg; i++) cf[i] = A.cof[i] - B.cof[i];
	return Polynomial<K>(cf, Deg, 0);
}

template <typename K>
std::ostream& operator<<(std::ostream& o, const Polynomial<K>& P) {
	o<<P.cof[0];
	for(int i = 1; i <= P.deg; i++) o<<" "<<P.cof[i];
	return o;
}

template <typename K>
std::vector<std::complex<long double> > Polynomial<K>::compute(
const std::vector<std::complex<long double> >& args) const {
	std::vector<std::complex<long double> > result;
	for(auto val : args)
		result.push_back(this->operator()(val));
	return result;
}

template <typename K>
std::vector<std::complex<long double> >  Polynomial<K>::DFT(
const std::vector<std::complex<long double> >& args, 
const Polynomial& P) {
	if(P.deg <= 8) return P.compute(args);
	int n = P.deg + 1;
	K* left = new K[n/2];
	K* right = new K[n/2];
	int It = -1;
	for(int i = 0; i < n; i += 2) {
		left[++It] = P.cof[i];
		right[It] = P.cof[i+1];
	}
	It = -1;
	Polynomial<K> L(left, n/2-1, 0);
	Polynomial<K> R(right, n/2-1, 0);
	std::vector<std::complex<long double> > args2;
	for(int i = 0; i < n/2; i++) args2.push_back(args[i]*args[i]);
	std::vector<std::complex<long double> > resultL = DFT(args2, L);
	std::vector<std::complex<long double> > resultR = DFT(args2, R);
	std::vector<std::complex<long double> > result;
	auto itL = resultL.begin();
	auto itR = resultR.begin();
	auto it  = args.begin();
	while(itL != resultL.end()) {
		result.push_back(*itL + (*it)*(*itR));
		itL++;
		itR++;
		it++;
	}
	itL = resultL.begin();
	itR = resultR.begin();
	while(itR != resultR.end()) {
		result.push_back(*itL + (*it)*(*itR));
		itL++;
		itR++;
		it++;
	}
	return result;
}

template <typename K>
std::vector<std::complex<long double> > Polynomial<K>::DFT(int n_min) 
const {
	int n = n_min;
	while(n <= deg) n *= 2;
	K* cof = new K[n];
	std::memcpy(cof, this->cof, (this->deg + 1)*sizeof(K) );
	K zero{};
	for(int i = deg+1; i<n; i++) cof[i] = zero;
	Polynomial<K> polynomial(cof, n-1, 0);
	std::vector<std::complex<long double> > args;
	for(int i=0; i<n; i++) args.push_back(std::polar(1.0, 2*i*M_PI/n));
	return DFT(args, polynomial);
}

template <typename K>
Polynomial<K> Polynomial<K>::mullN2(
const Polynomial<K>& A, const Polynomial<K>& B){
	int n = A.deg+B.deg;
	K* tmp = new K[n+1]();
	for(int i=0; i<=A.deg; i++){
		for(int j=0; j<=B.deg; j++){
			tmp[i+j] += A.cof[i]*B.cof[j];
		}
	}
	return Polynomial<K>(tmp, n, 0);
}

template <typename K>
template <typename T,
std::enable_if_t<std::is_integral<T>::value, bool> >
void Polynomial<K>::convert(
const std::complex<long double>& from, 
T& to) {
	to = std::round(std::real(from));
}


template <typename K>
template <typename T,
std::enable_if_t<std::is_floating_point<T>::value, bool> >
void Polynomial<K>::convert(const std::complex<long double>& from, 
T& to) {
	to = std::real(from);
}

template <typename K>
void Polynomial<K>::convert(const std::complex<long double>& from, 
std::complex<long double>& to) {
	to = from;
}

template <typename K>
void Polynomial<K>::convert(const std::complex<long double>& from, 
std::complex<double>& to) {
	to = std::complex<double>(std::real(from), std::imag(from));
}
template <typename K>
void Polynomial<K>::convert(const std::complex<long double>& from, 
std::complex<float>& to) {
	to = std::complex<float>(std::real(from), std::imag(from));
}

template <typename K>
template <typename T,
std::enable_if_t<!(std::is_integral<T>::value
|| std::is_floating_point<T>::value
), bool> >
void Polynomial<K>::convert(
const std::complex<long double>& from, 
T& to) {
	to = (T)from;
}

