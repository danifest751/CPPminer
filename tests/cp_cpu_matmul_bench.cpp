// Isolated, single-thread CPU experiment. No pool, proof, or miner entry point.
#include "pearl_matmul_algorithms.hpp"
#include "case33_gemm_xor.hpp"
#include "case33_cpu_features.hpp"
#include "cp_noise.h"
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace pearl_matmul;
using Clock=std::chrono::steady_clock;
namespace {
volatile uint64_t sink=0;
uint64_t checked_values=0, checked_cases=0, current_range_failures=0;
std::string quote(const std::string& s){
    std::string out="\"";
    for(char c:s){if(c=='"' || c=='\\')out+='\\';if(c=='\n')out+="\\n";else out+=c;}
    return out+'"';
}
void check(const std::vector<uint32_t>& got,const std::vector<uint32_t>& expected,const std::string& label){
    if(got.size()!=expected.size())throw std::runtime_error(label+": output size mismatch");
    for(size_t i=0;i<got.size();++i)if(got[i]!=expected[i])
        throw std::runtime_error(label+": mismatch at "+std::to_string(i));
    checked_values+=got.size();++checked_cases;
}
uint64_t checksum(const std::vector<uint32_t>& data){
    uint64_t h=1469598103934665603ULL;
    for(uint32_t x:data){h^=x;h*=1099511628211ULL;}
    return h;
}
struct Data {
    int m,n,k; Matrix<int16_t> a,bt;
    std::vector<int8_t> a8,b8;
    Data(int rows,int cols,int rank,int pattern,uint32_t seed):m(rows),n(cols),k(rank),a(m,k),bt(n,k),a8(size_t(m)*k),b8(size_t(n)*k){
        auto fill=[&](std::vector<int16_t>& dst,std::vector<int8_t>& dst8,bool b){
            for(size_t i=0;i<dst.size();++i){
                seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
                int v=0;
                if(pattern==1)v=b?-128:127;
                if(pattern==2)v=((i+i/k+b)&1)?127:-128;
                if(pattern==3)v=int(seed&127)-64;
                if(pattern==4)v=int(seed&255)-128;
                if(pattern==5)v=-128;
                dst[i]=int16_t(v);dst8[i]=int8_t(v);
            }
        };fill(a.data,a8,false);fill(bt.data,b8,true);
        if(pattern==6){
            uint8_t signal_seed[32],seed_a[32],seed_b[32];
            for(int i=0;i<32;++i){signal_seed[i]=uint8_t(i*11+17);seed_a[i]=uint8_t(i*7+3);seed_b[i]=uint8_t(i*13+9);}
            std::vector<int8_t> signal(a8.size());
            if(pearl_generate_random_a(signal_seed,32,m,k,signal.data())!=0 ||
               pearl_build_noisy_a(m,k,128,seed_a,signal.data(),a8.data())!=0 ||
               pearl_build_noisy_b(n,k,128,seed_b,nullptr,b8.data())!=0)
                throw std::runtime_error("zero-B fixture generation failed");
            std::copy(a8.begin(),a8.end(),a.data.begin());std::copy(b8.begin(),b8.end(),bt.data.begin());
        }
    }
};
std::vector<uint32_t> oracle(Data& d,int prefixes){
    std::vector<uint32_t> out(size_t(d.m)*d.n*prefixes);
    const int step=d.k/prefixes;
    // INT64, independent of every experimental and production microkernel.
    for(int i=0;i<d.m;++i)for(int j=0;j<d.n;++j){
        int64_t sum=0;
        for(int t=0;t<prefixes;++t){
            for(int k=t*step;k<(t+1)*step;++k)sum+=int64_t(d.a.view().row(i)[k])*d.bt.view().row(j)[k];
            out[size_t(t)*d.m*d.n+size_t(i)*d.n+j]=uint32_t(sum);
        }
    }return out;
}
void tile_xor(const uint32_t* c,int m,int n,uint32_t* out){
    int tile=0;
    for(int i=0;i<m;i+=8)for(int j=0;j<n;j+=16){
        uint32_t x=0;
        for(int r=0;r<8;++r)for(int s=0;s<16;++s)x^=c[size_t(i+r)*n+j+s];
        out[tile++]=x;
    }
}
std::vector<uint32_t> oracle_xor(const std::vector<uint32_t>& all,int m,int n){
    const size_t plane=size_t(m)*n,tiles=plane/128;
    std::vector<uint32_t> out(32*tiles);
    for(int t=0;t<32;++t)tile_xor(all.data()+t*plane,m,n,out.data()+t*tiles);
    return out;
}
struct Spec {std::string name;Kind kind;int depth;bool adaptive;};
std::vector<Spec> specs(bool vnni){
    std::vector<Spec> out={{"classical_compiler",Kind::Scalar,0,false},{"blocked_avx2_int16",Kind::Blocked16,0,false},
                           {"winograd_pairwise_avx2",Kind::Pairwise,0,false}};
    if(vnni)out.push_back({"classical_vnni_int8",Kind::Adaptive8,0,true});
    for(int depth=1;depth<=3;++depth){
        out.push_back({"strassen_d"+std::to_string(depth),Kind::Strassen,depth,false});
        out.push_back({"strassen_winograd_d"+std::to_string(depth),Kind::StrassenWinograd,depth,false});
    }
    if(vnni){out.push_back({"strassen_d1_adaptive",Kind::Strassen,1,true});
             out.push_back({"strassen_winograd_d1_adaptive",Kind::StrassenWinograd,1,true});}
    return out;
}
struct Pearl {
    Data& d; Engine engine; Matrix<uint32_t> delta,c;
    std::vector<uint32_t> xor_out;
    Pearl(Data& data,const Spec& spec):d(data),engine(spec.kind,spec.depth,spec.adaptive,d.m,d.n,128),
        delta(d.m,d.n),c(d.m,d.n),xor_out(size_t(d.m)*d.n/128*32){}
    void run(const std::vector<uint32_t>* expected=nullptr,Stats* stats=nullptr){
        std::fill(c.data.begin(),c.data.end(),0);
        const size_t plane=c.data.size(),tiles=plane/128;
        for(int t=0;t<32;++t){
            engine.multiply(input(d.a.view()).sub(0,t*128,d.m,128),input(d.bt.view()).sub(0,t*128,d.n,128),delta.view(),stats);
            for(size_t i=0;i<plane;++i)c.data[i]+=delta.data[i];
            if(expected){
                for(size_t i=0;i<plane;++i)if(c.data[i]!=(*expected)[t*plane+i])
                    throw std::runtime_error("Pearl C mismatch at milestone "+std::to_string(t)+", element "+std::to_string(i));
                checked_values+=plane;++checked_cases;
            }
            tile_xor(c.data.data(),d.m,d.n,xor_out.data()+t*tiles);
        }
    }
};
struct Method {
    std::string name,backend; std::function<void()> run;
    std::function<const std::vector<uint32_t>&()> output;
    std::vector<double> ms; std::vector<int> batches;
    size_t workspace=0; Stats stats;
};
double batch(Method& method,int count){
    auto start=Clock::now();
    for(int i=0;i<count;++i){method.run();asm volatile("" ::: "memory");}
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    sink=checksum(method.output());
    return ms;
}
void measure(std::vector<Method>& methods,const std::vector<uint32_t>& expected,
             const std::string& workload,int m,int n,int k,const std::string& pattern,int repeats,double target_ms){
    std::vector<int> order(methods.size());std::iota(order.begin(),order.end(),0);
    std::vector<int> batch_size(methods.size());
    for(size_t i=0;i<methods.size();++i){
        auto& method=methods[i];
        method.run();check(method.output(),expected,method.name);
        int count=1;double total=batch(method,count);
        while(total<target_ms && count<4096){count*=2;total=batch(method,count);}
        batch_size[i]=count;
    }
    // Rotate the initial order by the round number, then alternate direction.
    for(int r=0;r<repeats;++r){
        std::iota(order.begin(),order.end(),0);
        std::rotate(order.begin(),order.begin()+(r%order.size()),order.end());
        if(r&1)std::reverse(order.begin(),order.end());
        for(int i:order){
            double total=batch(methods[i],batch_size[i]);
            methods[i].ms.push_back(total/batch_size[i]);methods[i].batches.push_back(batch_size[i]);
            check(methods[i].output(),expected,methods[i].name+" timed");
        }
    }
    for(auto& x:methods){
        auto sorted=x.ms;std::sort(sorted.begin(),sorted.end());
        const double median=sorted.size()%2?sorted[sorted.size()/2]:(sorted[sorted.size()/2-1]+sorted[sorted.size()/2])/2;
        std::cout<<"{\"type\":\"measurement\",\"workload\":"<<quote(workload)<<",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k
                 <<",\"pattern\":"<<quote(pattern)<<",\"method\":"<<quote(x.name)<<",\"backend\":"<<quote(x.backend)
                 <<",\"median_ms\":"<<median<<",\"workspace_bytes\":"<<x.workspace
                 <<",\"leaf_macs\":"<<x.stats.leaf_macs<<",\"leaves_int8\":"<<x.stats.leaves_int8<<",\"leaves_int16\":"<<x.stats.leaves_int16
                 <<",\"max_intermediate_abs\":"<<x.stats.max_operand_abs<<",\"intermediate_values_outside_int8\":"<<x.stats.outside_int8
                 <<",\"checksum\":"<<quote(std::to_string(checksum(x.output())))<<",\"samples_ms\":[";
        for(size_t i=0;i<x.ms.size();++i){if(i)std::cout<<',';std::cout<<x.ms[i];}
        std::cout<<"],\"batch_counts\":[";
        for(size_t i=0;i<x.batches.size();++i){if(i)std::cout<<',';std::cout<<x.batches[i];}
        std::cout<<"]}"<<std::endl;
    }
}
void unit_tests(const std::vector<Spec>& methods){
    for(auto shape:std::vector<std::array<int,3>>{{3,5,7},{17,19,33},{32,64,128},{64,64,64},{128,128,128}})
        for(int pattern=0;pattern<=5;++pattern){
            Data d(shape[0],shape[1],shape[2],pattern,0x1234567u+pattern);auto ref=oracle(d,1);
            for(const auto& spec:methods){Engine e(spec.kind,spec.depth,spec.adaptive,d.m,d.n,d.k);Matrix<uint32_t> c(d.m,d.n);
                e.multiply(input(d.a.view()),input(d.bt.view()),c.view());check(c.data,ref,spec.name+" unit");}
        }
}
void plain_suite(int m,int n,int k,const std::vector<Spec>& specs,int repeats,double target_ms){
    Data d(m,n,k,4,0xabc123u);auto ref=oracle(d,1);
    struct State{Engine e;Matrix<uint32_t> c;State(const Spec& s,Data& d):e(s.kind,s.depth,s.adaptive,d.m,d.n,d.k),c(d.m,d.n){}};
    std::vector<std::unique_ptr<State>> states;std::vector<Method> methods;
    for(const auto& spec:specs){
        states.push_back(std::make_unique<State>(spec,d));auto* p=states.back().get();
        Method method;method.name=spec.name;method.backend="standalone";method.workspace=p->e.workspace_bytes();
        method.run=[p,&d]{p->e.multiply(input(d.a.view()),input(d.bt.view()),p->c.view());};method.output=[p]() -> const std::vector<uint32_t>& {return p->c.data;};
        p->e.multiply(input(d.a.view()),input(d.bt.view()),p->c.view(),&method.stats);check(p->c.data,ref,spec.name+" stats");
        methods.push_back(std::move(method));
    }
    measure(methods,ref,"final_gemm",m,n,k,"full_signed_int8",repeats,target_ms);
}
void pearl_suite(int size,int pattern,const std::vector<Spec>& specs,int repeats,double target_ms,bool verify_only){
    Data d(size,size,4096,pattern,0xf12345u);auto all=oracle(d,32),ref=oracle_xor(all,d.m,d.n);
    std::vector<std::unique_ptr<Pearl>> states;std::vector<std::unique_ptr<Case33GemmXor>> baselines;
    std::vector<Method> methods;
    for(auto isa:{Case33Isa::Auto,Case33Isa::Avx2,Case33Isa::Scalar}){
        baselines.push_back(std::make_unique<Case33GemmXor>());auto* p=baselines.back().get();p->set_isa(isa);
        if(!p->init(d.m,d.n,d.k,d.a8.data(),d.b8.data()))throw std::runtime_error("Case33 init failed");
        Method method;method.name=isa==Case33Isa::Auto?"current_case33_auto":isa==Case33Isa::Avx2?"current_case33_avx2":"current_case33_scalar";
        method.backend=p->backend();method.stats.leaf_macs=uint64_t(d.m)*d.n*d.k;
        method.run=[p]{p->run();};method.output=[p]() -> const std::vector<uint32_t>& {return p->tile_xor();};
        method.run();
        // Keep the observed full-range AVX2 saturation failure explicit. Every
        // other mismatch (including VNNI or generated zero-B data) is fatal.
        if(method.output()!=ref && pattern==4 && p->isa_used()==Case33Isa::Avx2){
            ++current_range_failures;
            size_t mismatches=0,first=0;
            for(size_t i=0;i<ref.size();++i)if(method.output()[i]!=ref[i]){if(!mismatches)first=i;++mismatches;}
            std::cout<<"{\"type\":\"current_backend_range_failure\",\"method\":"<<quote(method.name)
                     <<",\"backend\":"<<quote(method.backend)<<",\"m\":"<<d.m<<",\"n\":"<<d.n
                     <<",\"pattern\":\"full_signed_int8\",\"mismatched_tile_xors\":"<<mismatches
                     <<",\"first_index\":"<<first<<",\"got\":"<<method.output()[first]<<",\"expected\":"<<ref[first]
                     <<",\"timed\":false}"<<std::endl;
            continue;
        }
        check(method.output(),ref,method.name);methods.push_back(std::move(method));
        if(isa==Case33Isa::Auto){
            Method packed;packed.name="current_case33_auto_pack_and_scan";packed.backend=p->backend();
            packed.stats.leaf_macs=uint64_t(d.m)*d.n*d.k;
            packed.run=[p,&d]{
                if(!p->prepare_attempt_a(&d.a8,nullptr,nullptr,128))throw std::runtime_error("A pack failed");
                p->run();
            };
            packed.output=[p]() -> const std::vector<uint32_t>& {return p->tile_xor();};
            packed.run();check(packed.output(),ref,packed.name);methods.push_back(std::move(packed));
        }
    }
    for(const auto& spec:specs){
        states.push_back(std::make_unique<Pearl>(d,spec));auto* p=states.back().get();
        Method method;method.name=spec.name;method.backend="standalone";method.workspace=p->engine.workspace_bytes();
        method.run=[p]{p->run();};method.output=[p]() -> const std::vector<uint32_t>& {return p->xor_out;};
        p->run(&all,&method.stats);check(p->xor_out,ref,method.name);methods.push_back(std::move(method));
        p->run(&all);check(p->xor_out,ref,spec.name+" without stats");
    }
    if(!verify_only)measure(methods,ref,"pearl_32_prefixes",d.m,d.n,d.k,
                           pattern==6?"generated_zero_b":pattern==3?"signed_7_bit":"full_signed_int8",repeats,target_ms);
}
} // namespace
int main(int argc,char** argv){
    try{
        bool quick=false,verify_only=false;int repeats=7,cpu=2;double target_ms=40;
        for(int i=1;i<argc;++i){std::string arg=argv[i];
            if(arg=="--quick")quick=true;
            else if(arg=="--verify-only")verify_only=true;
            else if(arg=="--repeats" && i+1<argc)repeats=std::stoi(argv[++i]);
            else if(arg=="--sample-ms" && i+1<argc)target_ms=std::stod(argv[++i]);
            else if(arg=="--cpu" && i+1<argc)cpu=std::stoi(argv[++i]);
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        if(repeats<3 || repeats>31 || !std::isfinite(target_ms) || target_ms<10 || target_ms>500 || cpu<0 || cpu>63)
            throw std::runtime_error("repeats=3..31, sample-ms=10..500, cpu=0..63");
        omp_set_dynamic(0);omp_set_num_threads(1);
        bool pinned=false;
#ifdef _WIN32
        pinned=SetProcessAffinityMask(GetCurrentProcess(),DWORD_PTR(1)<<cpu)!=0;
#endif
        auto features=case33_detect_cpu_features();
        if(!features.avx2)throw std::runtime_error("this x86 experiment requires runtime AVX2 support");
        std::cout<<std::setprecision(10)<<"{\"type\":\"metadata\",\"compiler\":"<<quote(__VERSION__)
                 <<",\"threads\":1,\"affinity_cpu\":"<<(pinned?cpu:-1)<<",\"avx2\":true,\"avx512_vnni\":"<<(features.avx512_vnni?"true":"false")
                 <<",\"repeats\":"<<repeats<<",\"target_sample_ms\":"<<target_ms<<",\"quick\":"<<(quick?"true":"false")<<"}"<<std::endl;
        auto methods=specs(features.avx512_vnni);unit_tests(methods);
        if(!verify_only){
            plain_suite(128,128,128,methods,repeats,target_ms);
            plain_suite(256,256,256,methods,repeats,target_ms);
            if(!quick){plain_suite(512,512,512,methods,repeats,target_ms);plain_suite(128,128,4096,methods,repeats,target_ms);plain_suite(256,256,4096,methods,repeats,target_ms);}
        }
        pearl_suite(128,4,methods,repeats,target_ms,verify_only);
        if(!quick){
            pearl_suite(128,3,methods,repeats,target_ms,verify_only);pearl_suite(256,4,methods,repeats,target_ms,verify_only);
            pearl_suite(128,6,methods,repeats,target_ms,verify_only);pearl_suite(256,6,methods,repeats,target_ms,verify_only);
        }
        std::cout<<"{\"type\":\"validation\",\"passed\":true,\"checked_cases\":"<<checked_cases<<",\"checked_values\":"<<checked_values
                 <<",\"current_backend_range_failures\":"<<current_range_failures
                 <<",\"all_benchmarked_outputs_exact\":true,\"sink\":"<<quote(std::to_string(sink))<<"}"<<std::endl;
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}
}
