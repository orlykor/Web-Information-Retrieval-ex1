package webdata;

import java.util.*;


public class ReviewSearch {
    private IndexReader iReader;

    final Comparator<double[]> scoreComparator = new Comparator<double[]>() {
        @Override
        public int compare(double[] o1, double[] o2) {
            int res = Double.compare(o1[1], o2[1]);
            if (res == 0){
                return Double.compare(o2[0], o1[0]);
            }
            return res;
        }
    };
    final Comparator<String[]> scoreComparatorProductId = new Comparator<String[]>() {
        @Override
        public int compare(String[] o1, String[] o2) {
            int res = Double.compare(Double.parseDouble(o1[1]), Double.parseDouble(o2[1]));
            if (res == 0){
                return o2[0].compareTo(o1[0]);
            }
            return res;
        }
    };

    /**
     * Constructor
     */
    public ReviewSearch(IndexReader iReader) {
        this.iReader = iReader;
    }


    private HashMap<String, Integer> createTermWithFreq(Enumeration<String> query){
        String term;
        HashMap<String, Integer> queryTermsAndFreq = new HashMap<>();

        while (query.hasMoreElements()){
            term = query.nextElement().toLowerCase();
            if (!queryTermsAndFreq.containsKey(term)){
                queryTermsAndFreq.put(term, 1);
            }
            else {
                queryTermsAndFreq.put(term, queryTermsAndFreq.get(term)+1);
            }
        }
        return queryTermsAndFreq;
    }

    /**
     * Returns a list of the id-s of the k most highly ranked reviews for the
     * given query, using the vector space ranking function lnn.ltc (using the
     *
     SMART notation)
     * The list should be sorted by the ranking
     */
    public Enumeration<Integer> vectorSpaceSearch(Enumeration<String> query, int k) {
        double[] queryVector;
        HashMap<Integer, double[]> documentVectors = new HashMap<>();
        double[][] scores;
        int queryLength, docId, freqTermInQuery, numOfRew, freqTermInDoc;
        Enumeration<Integer> postingList;
        HashMap<String, Integer> queryTermsAndFreq;

        queryTermsAndFreq = createTermWithFreq(query);

        queryLength = queryTermsAndFreq.size();
        queryVector = new double[queryLength];
        int index = 0;
        double normFactor = 0, wt;
        for (String keyTerm : queryTermsAndFreq.keySet()) {
            //for the query vector
            freqTermInQuery = queryTermsAndFreq.get(keyTerm);
            numOfRew = iReader.getTokenFrequency(keyTerm);
            wt = this.computeQueryRank(freqTermInQuery, numOfRew);
            queryVector[index] = wt;
            normFactor += wt * wt;
            postingList = iReader.getReviewsWithToken(keyTerm);

            while(postingList.hasMoreElements()){
                docId = postingList.nextElement();
                if (!documentVectors.containsKey(docId)){
                    documentVectors.put(docId, new double[queryLength]);
                }
                freqTermInDoc = postingList.nextElement();
                documentVectors.get(docId)[index] = 1 + Math.log10(freqTermInDoc);

            }
            index++;
        }
        scores = new double[documentVectors.size()][2];

        if (normFactor != 0){
            normFactor = 1 / Math.sqrt(normFactor);
        }
        //normalize the vector of the query
        for (int i = 0; i < queryVector.length; i++){
            queryVector[i] = queryVector[i] * normFactor;
        }
        double scoreOfRew = 0;
        int j = 0;
        for (int key : documentVectors.keySet()){

            double[] docVector = documentVectors.get(key);
            for (int i = 0; i < queryVector.length; i++){
                scoreOfRew += queryVector[i] * docVector[i];
            }
            scores[j][0] = key;
            scores[j][1] = scoreOfRew;
            j++;
            scoreOfRew = 0;
        }
        return sortAndChooseK(scores, k);
    }

    private Enumeration<Integer> sortAndChooseK(double[][] scores, int k){
        Vector<Integer> chosenK = new Vector<>();
        Arrays.sort(scores, scoreComparator);

        for (int i = scores.length-1; i >= 0; i--){
            chosenK.add((int)scores[i][0]);
            if (i == scores.length - k){
                break;
            }
        }
        return chosenK.elements();
    }


    private double computeQueryRank(int freqTerm, int numOfRew) {
        double l, t, wt;
        int N = iReader.getNumberOfReviews();
        l = 1 + Math.log10(freqTerm);
        t = Math.log10(N/(double)numOfRew);
        wt = l * t;
        return wt;
    }



    /**
     * Returns a list of the id-s of the k most highly ranked reviews for the
     * given query, using the language model ranking function, smoothed using a
     * mixture model with the given value of lambda
     * The list should be sorted by the ranking
     */
    public Enumeration<Integer> languageModelSearch(Enumeration<String> query,
                                                    double lambda, int k) {
        HashMap<Integer, Double> queryProb = new HashMap<>();
        Enumeration<Integer> postingList;
        HashMap<Integer, Integer> hashPostingList;
        HashMap<String, Integer> queryTermsAndFreq;
        double[][] scores;
        int termFreqInDoc, termFreqInCollection, sizeOfCollection;
        double collectionProb, docProb, queryProbPerDoc, probRightSide = 1;

        queryTermsAndFreq = createTermWithFreq(query);
        sizeOfCollection = iReader.getTokenSizeOfReviews();
        for (String keyTerm : queryTermsAndFreq.keySet()){
            termFreqInCollection = iReader.getTokenCollectionFrequency(keyTerm);
            collectionProb = (1-lambda) * (termFreqInCollection / (double)sizeOfCollection);
            //for new docId that wasn't in the queryProb hashMap
            postingList = iReader.getReviewsWithToken(keyTerm);
            hashPostingList = new HashMap<>();

            while (postingList.hasMoreElements()){
                hashPostingList.put(postingList.nextElement(), postingList.nextElement());
            }

            for (int docId : queryProb.keySet()){
                if (hashPostingList.containsKey(docId)){
                    int lengthOfDoc = iReader.getReviewLength(docId);
                    termFreqInDoc = hashPostingList.get(docId);
                    docProb = lambda * termFreqInDoc / (double)lengthOfDoc;
                    queryProbPerDoc = Math.pow((docProb + collectionProb), queryTermsAndFreq.get(keyTerm));
                    queryProb.put(docId, queryProbPerDoc * queryProb.get(docId));
                    hashPostingList.remove(docId);
                }
                else {
                    queryProb.put(docId, collectionProb * queryProb.get(docId));
                }
            }
            for (int docId : hashPostingList.keySet()){
                int lengthOfDoc = iReader.getReviewLength(docId);
                termFreqInDoc = hashPostingList.get(docId);
                docProb = lambda * termFreqInDoc / (double)lengthOfDoc;
                queryProbPerDoc = Math.pow((docProb + collectionProb), queryTermsAndFreq.get(keyTerm));
                queryProb.put(docId, probRightSide * queryProbPerDoc);
            }
            probRightSide *= collectionProb;
        }
        int j = 0;
        scores = new double[queryProb.size()][2];
        for (int docId : queryProb.keySet()) {
            scores[j][0] = docId;
            scores[j][1] = queryProb.get(docId);
            j++;
        }
        return sortAndChooseK(scores, k);

    }



    /**
     * Returns a list of the id-s of the k most highly ranked productIds for the
     * given query using a function of your choice
     * The list should be sorted by the ranking
     */
    public Collection<String> productSearch(Enumeration<String> query, int k) {
        Enumeration<Integer> postingList, postingListProductId;
        HashMap<String, Integer> queryTermsAndFreq;
        String[][] scores;

        HashMap<String, Double> productIdScores = new HashMap<>();
        String productId;
        int docId, helpfulness;
        double score = 1;
        queryTermsAndFreq = createTermWithFreq(query);

        for(String term : queryTermsAndFreq.keySet()){
            postingList = iReader.getReviewsWithToken(term);
            while(postingList.hasMoreElements()){
                productId = iReader.getProductId(postingList.nextElement());
                postingList.nextElement();
                if (!productIdScores.containsKey(productId)){
                    postingListProductId = iReader.getProductReviews(productId);
                    while (postingListProductId.hasMoreElements()){
                        docId = postingListProductId.nextElement();
                        helpfulness = iReader.getReviewHelpfulnessDenominator(docId);
                        if (helpfulness == 0){
                            score *= iReader.getReviewScore(docId)/5.0;
                        }
                        else{
                            score *= (0.5 * (iReader.getReviewScore(docId)/5.0) +
                                    0.5 * (iReader.getReviewHelpfulnessNumerator(docId)/
                                            (double)iReader.getReviewHelpfulnessDenominator(docId)));
                        }
                    }
                    productIdScores.put(productId, score);
                }
            }

        }

        int j = 0;
        scores = new String[productIdScores.size()][2];
        for (String id : productIdScores.keySet()) {
            scores[j][0] = id;
            scores[j][1] = productIdScores.get(id).toString();
            j++;
        }


        Vector<String> chosenK = new Vector<>();
        Arrays.sort(scores, scoreComparatorProductId);

        for (int i = scores.length-1; i >= 0; i--){
            chosenK.add(scores[i][0]);
            if (i == scores.length - k){
                break;
            }
        }
        return chosenK;

    }
}
