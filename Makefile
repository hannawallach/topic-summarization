LIBS_DIR = libs
SRC_DIR = src
CORPORA_DIR = corpora
RESULTS_DIR = results

MAX_HEAP = 3g

JAVA_FLAGS = -server -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m
#JAVA_FLAGS = -server -enableassertions -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m

CP = $(LIBS_DIR)/mallet.jar:$(LIBS_DIR)/mallet-deps.jar

$(CORPORA_DIR)/%.dat: $(CORPORA_DIR)/csv/%.csv
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Csv2Vectors \
	--keep-sequence \
	--output $@ \
	--input $<

$(CORPORA_DIR)/%_no_stop.dat: $(CORPORA_DIR)/csv/%.csv
	cat $< | \
	sed 's/##NUMBER##//g' | \
	sed 's/ \+/ /g' > $(@D)/.%_no_stop.csv; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Csv2Vectors \
	--keep-sequence \
	--remove-stopwords \
	--extra-stopwords $(CORPORA_DIR)/stopwordlist.txt \
	--output $@ \
	--input $(@D)/.%_no_stop.csv; \
	rm $(@D)/.%_no_stop.csv

$(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID): $(CORPORA_DIR)/%.dat
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.topics.tui.Vectors2Topics \
	--input $< \
	--output-state $@/state.txt.gz \
	--output-topic-keys $@/topic-keys.txt \
	--xml-topic-report $@/topic-report.xml \
	--xml-topic-phrase-report $@/topic-phrase-report.xml \
	--output-doc-topics $@/doc-topics.txt \
	--num-topics $(T) \
	--num-iterations $(S) \
	--optimize-interval 5 \
	--optimize-burn-in 5 \
	> $@/stdout.txt 2>&1

$(RESULTS_DIR)/summaries/%/T$(T)-S$(S)-ID$(ID)_$(TEST)_$(SELECTION)_$(DIST)_L$(L)-C$(C).txt: $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)
	mkdir -p $(@D); \
	python -u $(SRC_DIR)/summarize.py \
	--state $</state.txt.gz \
	--topic-keys $</topic-keys.txt \
	--test $(TEST) \
	--selection $(SELECTION) \
	--dist $(DIST) \
	--max-phrase-len $(L) \
	--min-phrase-count $(C) \
	> $@

# example usage: for corpus in `echo AP_no_stop FOMC_no_stop NIPS_no_stop AP FOMC NIPS`; do for T in `echo 50 100`; do for test in `echo bayes-conditional chi-squared-yates`; do for selection in `echo none bigram n-1-gram`; do for dist in `echo average-posterior empirical prior`; do make results/summaries/"$corpus"/T"$T"-S5000-ID1_"$test"_"$selection"_"$dist"_L5-C5.txt T=$T S=5000 ID=1 TEST=$test SELECTION=$selection DIST=$dist L=5 C=5; done; done; done; done; done

$(RESULTS_DIR)/alt_summaries/%/T$(T)-S$(S)-ID$(ID)_$(DIST)_L$(L)-C$(C).txt: $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)
	mkdir -p $(@D); \
	python -u $(SRC_DIR)/alt_summarize.py \
	--state $</state.txt.gz \
	--topic-keys $</topic-keys.txt \
	--dist $(DIST) \
	--max-phrase-len $(L) \
	--min-phrase-count $(C) \
	> $@

$(RESULTS_DIR)/mallet_summaries/%/T$(T)-S$(S)-ID$(ID)_$(DIST)_L$(L)-C$(C).txt: $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)
	mkdir -p $(@D); \
	python -u $(SRC_DIR)/mallet_summarize.py \
	--state $</state.txt.gz \
	--topic-keys $</topic-keys.txt \
	--dist $(DIST) \
	--max-phrase-len $(L) \
	--min-phrase-count $(C) \
	> $@

$(RESULTS_DIR)/turbo_topics/%/T$(T)-S$(S)-ID$(ID)_no-perm_C$(C)-P$(P):  $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)/state.txt.gz
	mkdir -p $@; \
	python -u $(SRC_DIR)/turbo.py \
	--state $< \
	--output $@/corpus.txt \
	$@/vocab.txt \
	$@/assign.txt; \
	touch stop_words.txt; \
	tar zxvf $(LIBS_DIR)/turbotopics-py.tgz -C $(LIBS_DIR); \
	python -u $(LIBS_DIR)/turbotopics-py/lda_topics.py \
	--corpus=$@/corpus.txt \
	--vocab=$@/vocab.txt \
	--assign=$@/assign.txt \
	--ntopics=$(T) \
	--min-count=5 \
	--pval=$(P) \
	--out=$@/; \
	rm stop_words.txt

# example usage: for corpus in `echo AP_no_stop FOMC_no_stop NIPS_no_stop AP FOMC NIPS`; do for T in `echo 50 100`; do make results/turbo_topics/"$corpus"/T"$T"-S5000-ID1_no-perm_C5-P0.0001 T=$T S=5000 ID=1 C=5 P=0.0001; done; done

# note that this is what Dave suggested in person and faster than
# using permutation tests (as in the paper)

# for corpus in `echo AP AP_no_stop FOMC FOMC_no_stop NIPS NIPS_no_stop`; do python src/postprocess.py reformat --results results/lda/"$corpus"/T50-S5000-ID1/topic-keys.txt --output evaluation/"$corpus"/lda; done

# for corpus in `echo AP AP_no_stop FOMC FOMC_no_stop NIPS NIPS_no_stop`; do for dir in `echo alt_summaries mallet_summaries summaries`; do for x in `ls results/"$dir"/"$corpus"`; do python src/postprocess.py reformat --results results/"$dir"/"$corpus"/$x --output evaluation/"$corpus"/"$dir"; done; done; done

# for corpus in `echo AP AP_no_stop FOMC FOMC_no_stop NIPS NIPS_no_stop`; do python src/postprocess.py reformat --results results/turbo_topics/"$corpus"/T50-S5000-ID1_no-perm_C5-P0.0001/ --output evaluation/"$corpus"/turbo_topics; done

### compute statistics

# for corpus in `echo AP AP_no_stop FOMC FOMC_no_stop NIPS NIPS_no_stop`; do for dir in `echo alt_summaries mallet_summaries summaries`; do for x in `ls results/"$dir"/"$corpus"`; do python src/postprocess.py compute_stats --results results/"$dir"/"$corpus"/$x --output stats/"$corpus"/"$dir"; done; done; done

# for corpus in `echo AP AP_no_stop FOMC FOMC_no_stop NIPS NIPS_no_stop`; do python src/postprocess.py compute_stats --results results/turbo_topics/"$corpus"/T50-S5000-ID1_no-perm_C5-P0.0001/ --output stats/"$corpus"/turbo_topics; done
