// Standalone exact-integer experiments; not used by the production miner.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

namespace pearl_matmul {
template<class T> struct View {
    T* data; int rows, cols, stride;
    T* row(int i) const { return data + size_t(i)*stride; }
    View sub(int i,int j,int m,int n) const { return {row(i)+j,m,n,stride}; }
};
template<class T> struct Matrix {
    int rows,cols; std::vector<T> data;
    Matrix(int m,int n):rows(m),cols(n),data(size_t(m)*n){}
    View<T> view(){return {data.data(),rows,cols,cols};}
};
using Input=View<const int16_t>;
inline Input input(View<int16_t> x){return {x.data,x.rows,x.cols,x.stride};}

struct Stats {
    uint64_t leaf_macs=0, leaves_int8=0, leaves_int16=0;
    int max_operand_abs=0; uint64_t outside_int8=0;
};
inline void combine(Input a,Input b,View<int16_t> out,int sign,Stats* stats){
    for(int i=0;i<out.rows;++i) for(int j=0;j<out.cols;++j){
        int value=int(a.row(i)[j])+sign*int(b.row(i)[j]);
        // <=3 recursion levels and signed INT8 roots bound Winograd sums by 8192.
        if(value < -32768 || value > 32767) throw std::runtime_error("INT16 operand overflow");
        out.row(i)[j]=int16_t(value);
    }
    if(stats) for(int i=0;i<out.rows;++i) for(int j=0;j<out.cols;++j){
        int value=out.row(i)[j]; stats->max_operand_abs=std::max(stats->max_operand_abs,std::abs(value));
        stats->outside_int8 += value < -128 || value > 127;
    }
}

inline uint32_t dot_scalar(const int16_t* a,const int16_t* b,int k){
    uint32_t result=0;
    for(int l=0;l<k;++l) result+=uint32_t(int32_t(a[l])*int32_t(b[l]));
    return result;
}
__attribute__((target("avx2"))) inline uint32_t dot_avx2(const int16_t* a,const int16_t* b,int k){
    __m256i sum=_mm256_setzero_si256(); int l=0;
    for(;l+16<=k;l+=16)
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(
            _mm256_loadu_si256((const __m256i*)(a+l)),_mm256_loadu_si256((const __m256i*)(b+l))));
    alignas(32) uint32_t lanes[8]; _mm256_store_si256((__m256i*)lanes,sum);
    uint32_t result=0; for(auto value:lanes) result+=value;
    return result+dot_scalar(a+l,b+l,k-l);
}
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))) inline uint32_t dot_vnni(
    const uint8_t* a,const int8_t* b,int k,uint32_t compensation){
    __m512i sum=_mm512_setzero_si512(); int l=0;
    for(;l+64<=k;l+=64)
        sum=_mm512_dpbusd_epi32(sum,_mm512_loadu_si512(a+l),_mm512_loadu_si512(b+l));
    alignas(64) uint32_t lanes[16]; _mm512_store_si512(lanes,sum);
    uint32_t result=compensation; for(auto value:lanes) result+=value;
    for(;l<k;++l) result+=uint32_t(int32_t(a[l])*int32_t(b[l]));
    return result;
}
// Winograd's pair identity: (a_even+b_odd)*(a_odd+b_even) - row_factor - col_factor.
__attribute__((target("avx2"))) inline uint32_t pair_dot(const int16_t* a,const int16_t* b,int k){
    const __m256i even=_mm256_set1_epi32(0x0000ffff); __m256i sum=_mm256_setzero_si256(); int l=0;
    for(;l+16<=k;l+=16){
        __m256i av=_mm256_loadu_si256((const __m256i*)(a+l));
        __m256i bv=_mm256_loadu_si256((const __m256i*)(b+l));
        bv=_mm256_shufflehi_epi16(_mm256_shufflelo_epi16(bv,0xb1),0xb1);
        __m256i x=_mm256_add_epi16(av,bv);
        __m256i y=_mm256_shufflehi_epi16(_mm256_shufflelo_epi16(x,0xb1),0xb1);
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(_mm256_and_si256(x,even),y));
    }
    alignas(32) uint32_t lanes[8]; _mm256_store_si256((__m256i*)lanes,sum);
    uint32_t result=0; for(auto value:lanes) result+=value;
    for(;l+1<k;l+=2) result+=uint32_t((int32_t(a[l])+b[l+1])*(int32_t(a[l+1])+b[l]));
    if(l<k) result+=uint32_t(int32_t(a[l])*b[l]);
    return result;
}
inline uint32_t pair_factor(const int16_t* x,int k){
    uint32_t result=0;
    for(int l=0;l+1<k;l+=2) result+=uint32_t(int32_t(x[l])*x[l+1]);
    return result;
}

enum class Kind { Scalar, Blocked16, Adaptive8, Pairwise, Strassen, StrassenWinograd };
struct Frame {
    std::array<Matrix<int16_t>,4> a,b;
    std::array<Matrix<uint32_t>,7> p;
    std::vector<uint8_t> pack_a; std::vector<int8_t> pack_b;
    std::vector<uint32_t> compensation;
    Frame(int m,int n,int k):
        a{Matrix<int16_t>(m/2,k/2),Matrix<int16_t>(m/2,k/2),Matrix<int16_t>(m/2,k/2),Matrix<int16_t>(m/2,k/2)},
        b{Matrix<int16_t>(n/2,k/2),Matrix<int16_t>(n/2,k/2),Matrix<int16_t>(n/2,k/2),Matrix<int16_t>(n/2,k/2)},
        p{Matrix<uint32_t>(m/2,n/2),Matrix<uint32_t>(m/2,n/2),Matrix<uint32_t>(m/2,n/2),
          Matrix<uint32_t>(m/2,n/2),Matrix<uint32_t>(m/2,n/2),Matrix<uint32_t>(m/2,n/2),Matrix<uint32_t>(m/2,n/2)},
        pack_a(size_t(m)*k),pack_b(size_t(n)*k),compensation(n){}
};

class Engine {
    Kind kind_; int depth_; bool adaptive_; std::vector<Frame> frames_;
    void leaf(Input a,Input bt,View<uint32_t> c,int level,Stats* stats){
        Frame& f=frames_[level]; bool fits=adaptive_;
        if(fits){
            for(int i=0;i<a.rows && fits;++i) for(int k=0;k<a.cols;++k)
                if(a.row(i)[k]<-128 || a.row(i)[k]>127){fits=false;break;}
            for(int j=0;j<bt.rows && fits;++j) for(int k=0;k<bt.cols;++k)
                if(bt.row(j)[k]<-128 || bt.row(j)[k]>127){fits=false;break;}
        }
        if(fits){
            for(int i=0;i<a.rows;++i) for(int k=0;k<a.cols;++k)
                f.pack_a[size_t(i)*a.cols+k]=uint8_t(int(a.row(i)[k])+128);
            for(int j=0;j<bt.rows;++j){
                int32_t sum=0;
                for(int k=0;k<bt.cols;++k){int value=bt.row(j)[k];f.pack_b[size_t(j)*bt.cols+k]=int8_t(value);sum+=value;}
                f.compensation[j]=uint32_t(-128*sum);
            }
        }
        if(stats){stats->leaf_macs+=uint64_t(a.rows)*bt.rows*a.cols; if(fits)++stats->leaves_int8;else ++stats->leaves_int16;}
        // Fixed 16x16 output blocking. B is already stored transposed for contiguous dots.
        for(int i0=0;i0<a.rows;i0+=16) for(int j0=0;j0<bt.rows;j0+=16)
            for(int i=i0;i<std::min(i0+16,a.rows);++i) for(int j=j0;j<std::min(j0+16,bt.rows);++j){
                uint32_t value;
                if(fits) value=dot_vnni(f.pack_a.data()+size_t(i)*a.cols,f.pack_b.data()+size_t(j)*bt.cols,a.cols,f.compensation[j]);
                else if(kind_==Kind::Scalar) value=dot_scalar(a.row(i),bt.row(j),a.cols);
                else value=dot_avx2(a.row(i),bt.row(j),a.cols);
                c.row(i)[j]=value;
            }
    }
    void recurse(Input a,Input bt,View<uint32_t> c,int level,Stats* stats){
        if(level>=depth_ || a.rows%2 || bt.rows%2 || a.cols%2 || std::min({a.rows,bt.rows,a.cols})<16){leaf(a,bt,c,level,stats);return;}
        Frame& f=frames_[level]; int m=a.rows/2,n=bt.rows/2,k=a.cols/2;
        const auto a11=a.sub(0,0,m,k),a12=a.sub(0,k,m,k),a21=a.sub(m,0,m,k),a22=a.sub(m,k,m,k);
        // B12 is the bottom-left quadrant of its transposed storage.
        const auto b11=bt.sub(0,0,n,k),b12=bt.sub(n,0,n,k),b21=bt.sub(0,k,n,k),b22=bt.sub(n,k,n,k);
        auto s=[&](int i,Input x,Input y,int sign){combine(x,y,f.a[i].view(),sign,stats);return input(f.a[i].view());};
        auto t=[&](int i,Input x,Input y,int sign){combine(x,y,f.b[i].view(),sign,stats);return input(f.b[i].view());};
        auto product=[&](int i,Input x,Input y){recurse(x,y,f.p[i].view(),level+1,stats);};
        if(kind_==Kind::Strassen){
            product(0,s(0,a11,a22,1),t(0,b11,b22,1));
            product(1,s(0,a21,a22,1),b11);
            product(2,a11,t(0,b12,b22,-1));
            product(3,a22,t(0,b21,b11,-1));
            product(4,s(0,a11,a12,1),b22);
            product(5,s(0,a21,a11,-1),t(0,b11,b12,1));
            product(6,s(0,a12,a22,-1),t(0,b21,b22,1));
        }else{
            // Boyer/Dumas/Pernet/Zhou, arXiv:0707.2347, section 2.
            const auto s1=s(0,a21,a22,1),s2=s(1,s1,a11,-1),s3=s(2,a11,a21,-1),s4=s(3,a12,s2,-1);
            const auto t1=t(0,b12,b11,-1),t2=t(1,b22,t1,-1),t3=t(2,b22,b12,-1),t4=t(3,t2,b21,-1);
            product(0,a11,b11); product(1,a12,b21); product(2,s4,b22);
            product(3,a22,t4); product(4,s1,t1); product(5,s2,t2); product(6,s3,t3);
        }
        for(int i=0;i<m;++i) for(int j=0;j<n;++j){
            std::array<uint32_t,7> p; for(int t=0;t<7;++t)p[t]=f.p[t].view().row(i)[j];
            if(kind_==Kind::Strassen){
                c.row(i)[j]=p[0]+p[3]-p[4]+p[6]; c.row(i)[j+n]=p[2]+p[4];
                c.row(i+m)[j]=p[1]+p[3]; c.row(i+m)[j+n]=p[0]-p[1]+p[2]+p[5];
            }else{
                uint32_t u2=p[0]+p[5],u3=u2+p[6],u4=u2+p[4];
                c.row(i)[j]=p[0]+p[1]; c.row(i)[j+n]=u4+p[2];
                c.row(i+m)[j]=u3-p[3]; c.row(i+m)[j+n]=u3+p[4];
            }
        }
    }
public:
    Engine(Kind kind,int depth,bool adaptive,int m,int n,int k):kind_(kind),depth_(depth),adaptive_(adaptive){
        if(depth<0 || depth>3)throw std::runtime_error("depth must be 0..3");
        frames_.reserve(depth+1);
        for(int d=0;d<=depth;++d){frames_.emplace_back(m,n,k);m=(m+1)/2;n=(n+1)/2;k=(k+1)/2;}
    }
    size_t workspace_bytes()const{
        size_t bytes=0;for(const auto& f:frames_){
            for(const auto& x:f.a)bytes+=x.data.size()*2;
            for(const auto& x:f.b)bytes+=x.data.size()*2;
            for(const auto& x:f.p)bytes+=x.data.size()*4;
            bytes+=f.pack_a.size()+f.pack_b.size()+f.compensation.size()*4;
        }return bytes;
    }
    void multiply(Input a,Input bt,View<uint32_t> c,Stats* stats=nullptr){
        if(kind_==Kind::Pairwise){
            auto& f=frames_[0];
            for(int j=0;j<bt.rows;++j)f.compensation[j]=pair_factor(bt.row(j),bt.cols);
            for(int i=0;i<a.rows;++i){uint32_t row=pair_factor(a.row(i),a.cols);
                for(int j=0;j<bt.rows;++j)c.row(i)[j]=pair_dot(a.row(i),bt.row(j),a.cols)-row-f.compensation[j];}
            if(stats)stats->leaf_macs+=uint64_t(a.rows)*bt.rows*((a.cols+1)/2)+uint64_t(a.rows+bt.rows)*(a.cols/2);
        }else recurse(a,bt,c,0,stats);
    }
};
} // namespace pearl_matmul
