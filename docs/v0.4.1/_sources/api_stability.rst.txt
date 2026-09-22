API stability policy
====================

DeepPeak follows semantic versioning. Public names documented in the API
reference are covered by the following policy:

* patch releases fix defects without intentional public API changes;
* minor releases may add backward-compatible functionality;
* removals and incompatible signature changes require a major release;
* a deprecation normally remains for at least one minor release and emits a
  ``DeprecationWarning`` with its replacement;
* modules, names, and attributes beginning with an underscore are private.

Experimental APIs are explicitly labelled and may change in a minor release.
Saved neural weights are only guaranteed to load with the documented DeepPeak
and TensorFlow version range in their model card.
