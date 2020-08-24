
import numpy as np
import sys


class TN(object):
    def __init__(self, D, I, TEMP_WND=3, MIN_MATCH=3):
        sys.setrecursionlimit(5000)

        self.TEMP_WND = TEMP_WND
        self.MIN_MATCH = MIN_MATCH

        self.index = I
        self.dist = D

        self.query_length = D.shape[0]
        # isdetect, next_time,next_rank, scores
        self.paths = np.zeros((*D.shape, 4))

        # # (next_time, next_rank, max_query_time, max_ref_time, match_cnt)
        # self.dy_table = np.ones((self.query_time, self.TOP_K, 5), dtype=np.int) * -1
        # # path score
        # self.dy_score = np.zeros((self.query_time, self.TOP_K), dtype=np.float)

    def find_linkable_node(self, t, r):

        v_id, f_id = self.index[t, r]
        time, rank = np.where((self.index[t + 1:t + 1 + self.TEMP_WND][0] == v_id) &
                              (f_id <= self.index[t + 1:t + 1 + self.TEMP_WND][1]) &
                              (self.index[t + 1:t + 1 + self.TEMP_WND][1] <= f_id + self.TEMP_WND))
        print(v_id,f_id,time,rank,self.TEMP_WND,t,r)

        return np.dstack((time + 1 + t, rank)).squeeze(0).tolist()

    def find_max_score_path(self, t, r):
        print(t,r)
        if self.paths[t, r, 0] != 1:
            nodes = self.find_linkable_node(t, r)
            print(nodes)
            paths = [[time, rank, *self.find_max_score_path(time, rank)] for time, rank in nodes]
            if len(paths) != 0:
                path = sorted(paths, key=lambda x: x[-1], reverse=True)[0]

                next_time, next_rank = path[0],path[1]
                score = path[-1] + self.dist[t, r]
            else:
                next_time, next_rank, score = -1, -1, self.dist[t, r]

            self.paths[t, r] = [1, next_time, next_rank, score]

        print(t,r,self.paths[t,r])
        return self.paths[t, r]

    def fit(self):
        candidate = []
        for t in range(self.query_length):
            for rank, (v_idx, f_idx) in enumerate(self.index[t]):
                print('fit',t,rank)
                self.find_max_score_path(t,rank)
        #
        # for time, tlist in enumerate(self.dist.shape[0]):
        #     for rank, s in enumerate(tlist):
        #         if self.score[time, rank] > self.SCORE_THR and self.dy_table[time, rank][0] == -1:
        #             _, _, max_q, max_r, cnt, path_score = self.get_maximum_path(time, rank)
        #
        #             # half-closed form [ )
        #             if cnt >= self.MIN_MATCH:
        #                 detect = {'query': Period(time, max_q + 1),
        #                           'ref': Period(self.idx[time, rank], max_r + 1),
        #                           'match': cnt,
        #                           'score': path_score}
        #                 candidate.append(detect)

        # candidate.sort(key=lambda x: x['score'], reverse=True)
        # [print(c) for c in candidate]
        # 1. overlap -> NMS
        '''
        nms = [True for i in range(len(candidate))]
        for i in range(0, len(candidate)):
            key = candidate[i]
            for j in range(i + 1, len(candidate)):
                if nms[j] and key['query'].is_overlap(candidate[j]['query']) and key['ref'].is_overlap(
                        candidate[j]['ref']):
                    nms[j] = False
        sink = [candidate[i] for i in range(0, len(candidate)) if nms[i]]
        '''
        # 2. overlap -> overlay

        # 3. return maximum score
        sink = [] if not len(candidate) else [max(candidate, key=lambda x: x['score'])]

        # 4. return all candidate
        # sink = sorted(candidate, key=lambda x: x['score'])

        return sink

    def get_maximum_path(self, time, rank):
        if self.dy_table[time, rank][0] == -1:
            neighbor = self.get_neighbor(time, rank)
            # print(time, rank, neighbor)
            neighbor_paths = []
            for t, r in neighbor:
                _, _, max_q, max_r, cnt, path_score = self.get_maximum_path(t, r)
                neighbor_paths.append([t, r, max_q, max_r, cnt, path_score])

            # 이웃 o -> find max/ 이웃 x -> end path            
            if len(neighbor_paths):
                # longest path -> query 쪽 길이가 김
                max_neighbor = max(neighbor_paths, key=lambda x: x[4])
                self.dy_table[time, rank] = max_neighbor[:5]
                self.dy_table[time, rank, 4] += 1
                self.dy_score[time, rank] = max_neighbor[5] + self.score[time, rank]

            else:
                self.dy_table[time, rank] = [0, 0, time, self.idx[time, rank], 1]
                self.dy_score[time, rank] = self.score[time, rank]

        next_time, next_rank, max_query, max_ref, match_cnt = self.dy_table[time, rank]
        score = self.dy_score[time, rank]
        return next_time, next_rank, max_query, max_ref, match_cnt, score

    def get_neighbor(self, time, rank):
        curr_idx = self.idx[time, rank]
        neighbor = []
        min_idx = self.ref_time
        # next_t 에서는 현재 t 에서 추가한 이웃보다 작은 ts를 갖는 노드만 추가(중복 제거)
        # 큰 ts에 바로 연결할경우 sub-optimal path(동일한 path 이지만 score가 작음)
        # print(range(time + 1, min(time + self.TEMP_WND + 1, self.query_time)),min_idx)
        for t in range(time + 1, min(time + self.TEMP_WND + 1, self.query_time)):
            n_idx = []
            for r, idx in enumerate(self.idx[t]):
                if self.score[
                    t, rank] > self.SCORE_THR and curr_idx < idx <= curr_idx + self.TEMP_WND and idx <= min_idx:
                    neighbor.append((t, r))
                    n_idx.append(idx)
            min_idx = min_idx if not len(n_idx) else min(n_idx)
        return neighbor


if __name__ == '__main__':
    n = 5
    score = (np.arange(0, n ** 2) / n ** 2)[::-1].reshape(n, n)
    idx = np.concatenate([np.arange(0, n), np.arange(0, n), np.arange(0, n), np.arange(0, n), np.arange(0, n)]).reshape(
        n, n)

    print(score)
    print(score[3][2])
    print(score[3, 2])
    # print([list(s[s > 0.8]) for s in score])

    # tn = TN2(score, idx, TOP_K=3)
