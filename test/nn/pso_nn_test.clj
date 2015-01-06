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
    (position-to-nn (get-position (pso-nn t-in t-out structure 0.01 40 0.05 1000 :chart? true)) structure)))

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
  (let [network (iris-trainer '(5 10 10 1))]
    (println network)
    (iris-predictor network)))
