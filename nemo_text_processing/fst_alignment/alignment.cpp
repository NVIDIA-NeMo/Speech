//Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//  http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.


#include <fstream>
#include <fst/fstlib.h>
#include <thrax/thrax.h>
#include <string>
#include <cctype>

using fst::StdArcLookAheadFst;
using thrax::GrmManager;
using fst::StdArc;
using fst::StdFst;
using namespace std;
typedef StdArcLookAheadFst LookaheadFst;


// This file takes 1. FST path 2. entire string 3. start position of substring 4. end (exclusive) position of substring
// and prints 1. mapped output string 2. start and end indices of mapped substring

// Usage: 

// g++ -std=gnu++11 -I<path to env>/include/ alignment.cpp -lfst -lthrax -ldl -L<path to env>/lib 
// ./a.out <fst file> "tokenize_and_classify" "2615 Forest Av, 1 Aug 2016" 22 26

// Output:
// inp string: |2615 Forest Av, 1 Aug 2016|
// out string: |twenty six fifteen Forest Avenue , the first of august twenty sixteen|
// inp indices: [22:26]
// out indices: [55:69]
// in: |2016| out: |twenty sixteen|

// Disclaimer: The heuristic algorithm relies on monotonous alignment and can fail in certain situations,
// e.g. when word pieces are reordered by the fst, e.g. 

// ./a.out <fst file> "tokenize_and_classify" "$1" 0 1
// inp string: |$1|
// out string: |one dollar|
// inp indices: [0:1] out indices: [0:3]
// in: |$| out: |one|

// to prevent compiler error, not actually called
namespace fst {
  #include <fst/extensions/far/stlist.h>
  #include <fst/extensions/far/sttable.h>
  #include <cstdint>
  #include <ios>

  bool IsSTList(const std::string &source) {
    std::ifstream strm(source, std::ios_base::in | std::ios_base::binary);
    if (!strm) return false;
    int32_t magic_number = 0;
    ReadType(strm, &magic_number);
    return magic_number == kSTListMagicNumber;
  }

  bool IsSTTable(const std::string &source) {
    std::ifstream strm(source);
    if (!strm.good()) return false;

    int32_t magic_number = 0;
    ReadType(strm, &magic_number);
    return magic_number == kSTTableMagicNumber;
  }
} 

char EPS = '\0'; 
char WS= ' ';   


int _get_aligned_index(const vector<tuple<char, char>> &alignment, int index){
    int aligned_index = 0;

    int idx = 0;
    while (idx < index){
        if (get<0>(alignment[aligned_index]) != EPS) {idx += 1;}
        aligned_index += 1;
    }
    while (get<0>(alignment[aligned_index]) == EPS){
        aligned_index += 1;
    }
    return aligned_index;
}

int _get_original_index(const vector<tuple<char, char>> &alignment, int aligned_index){
    int og_index = 0;
    int idx = 0;
    while (idx < aligned_index) {
        if (get<1>(alignment[idx]) != EPS)  {og_index += 1;}
        idx += 1;
    }
    return og_index;
}


tuple<int, int> indexed_map_to_output(const vector<tuple<char, char>> &alignment, int start, int end) {

    int aligned_start = _get_aligned_index(alignment, start);
    int aligned_end = _get_aligned_index(alignment, end-1);

    string output_str = "";
    string input_str = "";
    for (const auto &i: alignment){
        output_str += get<1>(i);
        input_str += get<0>(i);
    }


    while ((aligned_start -1 > 0) && (get<0>(alignment[aligned_start-1]) == EPS) && (isalpha(get<1>(alignment[aligned_start-1])) || (get<1>(alignment[aligned_start-1]) == EPS))) {aligned_start -= 1;}

    while (((aligned_end + 1) < alignment.size()) && (get<0>(alignment[aligned_end + 1]) == EPS) && (isalpha(get<1>(alignment[aligned_end + 1])) || (get<1>(alignment[aligned_end + 1]) == EPS))) {aligned_end += 1;}

    while (((aligned_end + 1) < alignment.size()) && (isalpha(get<1>(alignment[aligned_end + 1])) || get<1>(alignment[aligned_end + 1]) == EPS)) {aligned_end += 1;}

    int output_og_start_index = _get_original_index(alignment, aligned_start);
    int output_og_end_index = _get_original_index(alignment, aligned_end+1);
    
    return make_tuple(output_og_start_index, output_og_end_index);


}

int main(int argc, char * argv[]) {
  if (argc != 6)
  {
    printf("\n Please provide 4 input arguments, 1st is the path to fst, 2nd the fst rule name, "
    "3rd the full string in quotes, 4th is start index of word, 5th is exclusive end index of word.");
    return 1;
  }
  unique_ptr<GrmManager> grm_;
  grm_.reset(new GrmManager);
  string grm_file=argv[1];
  string grm_rule=argv[2];
  grm_->LoadArchive(grm_file);
  LookaheadFst fst_(*(grm_->GetFst(grm_rule)));

  
  string input = argv[3];
  typedef fst::StringCompiler<StdArc> Compiler;
  typedef StdArc::StateId StateId;
  typedef fst::StringPrinter<StdArc> Printer;
  Compiler compiler;
  thrax::GrmManager::MutableTransducer input_fst;
  compiler(input, &input_fst);
  
  fst::ComposeFst<StdArc> output(input_fst,fst_);
  thrax::GrmManager::MutableTransducer shortest_path;
  fst::ShortestPath(output, &shortest_path);

  auto siter = shortest_path.Start();
  vector<tuple<char, char>> alignment; 

  while (shortest_path.Final(siter) == StdArc::Weight::Zero()) {
    fst::ArcIterator<StdFst> aiter(shortest_path, siter);
    if (aiter.Done()) {
      cerr << "Unexpected format: Does not reach final state" << endl;
      return 1;
    }
    const auto &arc = aiter.Value();
    alignment.push_back({(char)arc.ilabel, (char)arc.olabel });
    siter = arc.nextstate;
    aiter.Next();
    if (!aiter.Done()) {
      cerr << "Unexpected format: State has multiple outgoing arcs" << endl;
      return 1;
    }
  }

  string output_str = "";
  int idx = 0;
  for (const auto &i: alignment){
    if (get<1>(i) == EPS) continue; 
    output_str += get<1>(i);
  }

  int start_index = atoi(argv[4]);
  int end_index = atoi(argv[5]);
  
  tuple<int, int>  out_indices = indexed_map_to_output(alignment, start_index, end_index); 
  cout << "inp string: |" << argv[3] << "|" << endl;
  cout << "out string: |" << output_str << "|" << endl;
  cout << "inp indices: [" << start_index << ":" << end_index << "]" << endl;
  cout << "out indices: [" << get<0>(out_indices) << ":" << get<1>(out_indices) << "]" << endl;
  cout << "in: |" << input.substr(start_index, end_index - start_index) << "| out: |" << output_str.substr(get<0>(out_indices), (get<1>(out_indices)-get<0>(out_indices))) << "|" << endl;
}