from flwr.serverapp.strategy import FedAvg

class SixRegionFedAvg(FedAvg):
    def aggregate_train(self,server_round,replies):
        replies=list(replies)
        valid,failures=self._check_and_log_replies(replies,is_train=True)
        if failures: raise RuntimeError(f"Round {server_round}: client failures: {failures}")
        regions=[int(m.content["metrics"]["region-code"]) for m in valid]
        expected={1,2,3,4,5,6}
        if len(regions)!=6 or set(regions)!=expected or len(set(regions))!=6:
            raise RuntimeError(f"Round {server_round}: expected one reply per region; got {regions}")
        return super().aggregate_train(server_round,valid)
