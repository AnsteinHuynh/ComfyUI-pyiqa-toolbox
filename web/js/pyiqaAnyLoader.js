/**
 * pyiqa-toolbox: "(Down)load pyiqa Model" (PyiqaAnyLoader) dynamic variant combo.
 *
 * The variant dropdown's values are architecture-prefixed ("LPIPS | AlexNet
 * backbone - FR, lower better, 0-1"). Python lists ALL of them statically
 * (headless / API / extension-less clients stay fully functional); this
 * extension filters the widget's option list down to the selected architecture
 * whenever it changes, so the user only ever sees that architecture's entries
 * (its registry variants, then its custom checkpoints).
 *
 * ComfyUI combos cannot depend on each other, which is why the mapping travels
 * inside the values themselves instead of in a second request.
 *
 * - onNodeCreated: hook the architecture widget's callback + initial sync
 * - afterConfigureGraph: re-sync every loader after a workflow load
 *   (restoring widgets_values does NOT fire callbacks)
 * - if the current variant value is still valid for the new architecture it is
 *   kept (workflow re-loads keep their selection); otherwise it resets to the
 *   first entry of the new architecture
 */

import { app } from "../../scripts/app.js";

const CLASS_NAME = "PyiqaAnyLoader";
const SEP = " | ";
// per-node memory of the full, unfiltered value list (the widget's own
// options.values gets filtered in place, so keep the original aside)
const fullValues = new WeakMap();

function syncVariant(node) {
    if (!node?.widgets) return;
    const arch = node.widgets.find((w) => w.name === "architecture");
    const variant = node.widgets.find((w) => w.name === "variant");
    const values = variant?.options?.values;
    if (!arch || !variant || !Array.isArray(values)) return;

    let all = fullValues.get(node);
    if (!all || values.length > all.length) {
        // first sync, or the widget was re-registered with a longer list than
        // the one we cached: (re)capture the unfiltered values
        all = values.filter((v) => typeof v === "string");
        fullValues.set(node, all);
    }

    const prefix = String(arch.value) + SEP;
    const filtered = all.filter((v) => v.startsWith(prefix));
    if (filtered.length === 0) {
        // unknown architecture (catalog drift): fall back to the full list
        // rather than stranding the widget with an empty dropdown
        variant.options.values = all;
        return;
    }
    variant.options.values = filtered;
    if (!filtered.includes(variant.value)) {
        variant.value = filtered[0];
        try {
            variant.callback?.(variant.value);
        } catch (err) {
            console.warn("[pyiqa-toolbox] variant callback failed", err);
        }
    }
}

app.registerExtension({
    name: "pyiqa-toolbox.anyLoader",
    beforeRegisterNodeDef(nodeType, data) {
        if (data?.name !== CLASS_NAME) return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = orig?.apply(this, arguments);
            const node = this;
            const arch = node.widgets?.find((w) => w.name === "architecture");
            if (arch) {
                const origCallback = arch.callback;
                arch.callback = function () {
                    const r = origCallback?.apply(this, arguments);
                    syncVariant(node);
                    return r;
                };
            }
            // widgets may still be settling; sync once now and once deferred
            syncVariant(node);
            Promise.resolve().then(() => syncVariant(node));
            return result;
        };
    },
    afterConfigureGraph() {
        for (const node of app.graph?.nodes ?? []) {
            if (node?.type === CLASS_NAME || node?.comfyClass === CLASS_NAME) {
                syncVariant(node);
            }
        }
    },
    loadedGraphNode(node) {
        if (node?.type === CLASS_NAME || node?.comfyClass === CLASS_NAME) {
            syncVariant(node);
        }
    },
});
