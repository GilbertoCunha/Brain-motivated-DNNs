run:
	python3 -m RetinaVVS.RetinaVVS_models
	python3 -m SIFT.SIFT_models
	python3 -m LBP.LBP_models
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models
	make clean

test:
	python3 -m RetinaVVS.RetinaVVS_models --fast_dev_run --no-auto_lr_find
	python3 -m SIFT.SIFT_models --fast_dev_run --no-auto_lr_find
	python3 -m LBP.LBP_models --fast_dev_run --no-auto_lr_find
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models --fast_dev_run --no-auto_lr_find
	make clean

test_RetinaVVS:
	python3 -m RetinaVVS.RetinaVVS_models --fast_dev_run --no-auto_lr_find
	make clean

test_SIFT:
	python3 -m SIFT.SIFT_models --fast_dev_run --no-auto_lr_find
	make clean

test_LBP:
	python3 -m LBP.LBP_models --fast_dev_run --no-auto_lr_find
	make clean

test_Graph:
	python3 -m RetinaVVSGraph.RetinaVVSGraph_models --fast_dev_run --no-auto_lr_find
	make clean

clean:
	rm -r Models
	mkdir Models