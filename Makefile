run:
	python3 -m RetinaVVS.RetinaVVS_models
	python3 -m SIFT.SIFT_models
	python3 -m LBP.LBP_models
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models

test:
	python3 -m RetinaVVS.RetinaVVS_models --fast_dev_run --no-auto_lr_find
	python3 -m SIFT.SIFT_models --fast_dev_run --no-auto_lr_find
	python3 -m LBP.LBP_models --fast_dev_run --no-auto_lr_find
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models --fast_dev_run --no-auto_lr_find

test_RetinaVVS:
	python3 -m RetinaVVS.RetinaVVS_models --fast_dev_run --no-auto_lr_find

test_SIFT:
	python3 -m SIFT.SIFT_models --fast_dev_run --no-auto_lr_find

test_LBP:
	python3 -m LBP.LBP_models --fast_dev_run --no-auto_lr_find

test_Graph:
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models --fast_dev_run --no-auto_lr_find