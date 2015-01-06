(ns nn.pso-nn-test
  (:require [clojure.test :refer :all]
            [nn.pso-nn :refer :all]
            [clojure.string :as s]
            [nn.core :as n]))

;margin is relative to x (the steady value) and not y the approximate value
; (- x (* x margin)) < y < (+ x (* x margin))
(defn approx-equal [x y margin]
  (and (< (- x (* x margin)) y)
       (< y (+ x (* x margin)))))

(defn format-csv-data [filename]
  (loop [data (map #(s/split % #",") (s/split (slurp filename) #"\n"))
         t-in []
         t-out []]
    (if (empty? data)
      [t-in t-out]
      (let [elem (first data)]
        (recur (rest data)
               (conj t-in (cons 1 (map #(Float/parseFloat %) (drop-last elem))))
               (conj t-out
                     (let [l (Integer/parseInt (last elem))]
                       (cond (= l 1) [0]
                             (= l 2) [0]
                             (= l 3) [1]))))))))

(defn iris-trainer [structure]
  (let [[t-in t-out] (format-csv-data "iris-data.csv")]
    (position-to-nn (get-position (pso-nn t-in t-out structure 0.1 60 0.1 2000)) structure)))

(defn iris-predictor [network]
  (let [[in expected] (format-csv-data "iris-test-data.csv")]
    (loop [in in
           expected expected]
      (if (empty? in)
        nil
        (do
          (println "Expected:" (first expected))
          (println "Actual:" (n/hypothesis (first in) network))
          (recur (rest in) (rest expected)))))))

(defn iris-test []
  (let [network (iris-trainer '(5 20 1))]
    (println network)
    (iris-predictor network)))

;a working network for class 1 vs all classifier (5 10 1) structure
;[((-0.05287251447113406 0.4287280380277156 0.10289263716594571 -0.5302332614189089 -0.8476821109606543)
;  (0.15089695279337423 -0.49231005098487635 0.011661081352305036 0.3363190487424353 0.06558551444787653)
;  (0.23386106335752793 -0.46555526409200726 -0.7005517878867804 0.4497092301759215 0.6767595903383632)
;  (-0.7199686918482859 0.6585502285348527 -0.2769886523804212 -0.48073202368680745 -0.8522004961234544)
;  (-1.0020424805229626 0.09985127872234395 0.34123238860699656 0.6831923776220177 0.5810505105313728)
;  (-0.7579882289652113 -0.05546416263287729 -0.2592551763317205 0.41113076316080305 -0.6711999159394797)
;  (-0.7332511938400889 -0.7931930584855056 -0.11606905019846808 -0.15988706133079073 -0.6572765027180469)
;  (-0.008059122900934468 0.1183251443397676 0.24315421560446904 -0.49945192020378826 -0.40670152852674)
;  (0.7866657096153671 -0.3158616110234712 -0.1540459766083156 0.38324524291524026 0.3916617386656449)
;  (0.7125840777043808 -0.9024310085368092 -1.0979218147479126 -0.5471736361047821 0.007694273494428472))
; ((-0.5156657667433732 -0.18010857364129246 0.47226000608144675 0.5338411325651917 0.7059572289899372
;   -0.11452379187789374 0.3468979890975581 -0.9746834451564383 0.8874435323816507 -0.9014013812542323))]
