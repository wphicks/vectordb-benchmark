from utils.util_log import log


class InterfaceBase:
    @staticmethod
    def get_recall_value(true_ids, result_ids):
        """
        Use the intersection length
        """
        sum_radio = 0.0
        top_k_check = True
        for index, item in enumerate(result_ids):
            tmp = set(true_ids[index]).intersection(set(item))
            if len(item) != 0:
                sum_radio += len(tmp) / len(item)
            else:
                top_k_check = False
                log.error("[InterfaceBase] Length of returned top_k is 0, please check.")
        if top_k_check is False:
            raise ValueError("[InterfaceBase] The result of top_k is wrong, please check: {}".format(result_ids))
        return round(sum_radio / len(result_ids), 3)
